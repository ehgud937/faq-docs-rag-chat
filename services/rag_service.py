from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config.settings import settings
from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService
import logging
import math

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 파이프라인 통합 서비스"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.faiss_service = FAISSService()
        
        # GMS 클라이언트
        self.client = AsyncOpenAI(
            api_key=settings.GMS_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        
        # Reranker 모델 로드 (선택적)
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✅ Reranker 모델 로드 완료")
        except ImportError:
            logger.warning("⚠️ sentence-transformers 미설치 - Reranking 비활성화")
            logger.warning("   설치: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"⚠️ Reranker 로드 실패 (선택 기능): {e}")
        
        # 인덱스 자동 로드
        try:
            self.faiss_service.load_index()
        except FileNotFoundError:
            logger.warning("⚠️ FAISS 인덱스 없음 - 먼저 인덱스를 생성하세요")
    
    def _calculate_similarity(self, distance: float) -> float:
        """
        L2 distance를 유사도 점수(0~1)로 변환
        
        Args:
            distance: FAISS L2 distance
        
        Returns:
            유사도 점수 (0~1, 높을수록 유사)
        """
        # 지수 감쇠 함수
        similarity = math.exp(-distance / 2.0)
        return similarity
    
    def _keyword_search(self, query: str, k: int = 20) -> List[tuple]:
        """
        키워드 기반 검색 (기술 용어 매칭에 효과적)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
        
        Returns:
            [(metadata, keyword_score), ...]
        """
        results = []
        query_lower = query.lower()
        
        # 쿼리를 단어로 분리 (공백, 쉼표 등)
        import re
        query_terms = set(re.findall(r'\w+', query_lower))
        
        for metadata in self.faiss_service.metadata:
            # 질문, 답변, 키워드에서 매칭 점수 계산
            question = metadata.get('question', '').lower()
            answer = metadata.get('answer', '').lower()
            keywords = ' '.join(metadata.get('keywords', [])).lower()
            category = metadata.get('category', '').lower()
            
            # 가중치 부여 (질문 > 키워드 > 카테고리 > 답변)
            score = 0
            for term in query_terms:
                if len(term) < 2:  # 너무 짧은 단어 제외
                    continue
                
                # 질문에 있으면 높은 점수
                if term in question:
                    score += 10 * question.count(term)
                
                # 키워드에 있으면 중간 점수
                if term in keywords:
                    score += 5 * keywords.count(term)
                
                # 카테고리에 있으면
                if term in category:
                    score += 3
                
                # 답변에 있으면 낮은 점수
                if term in answer:
                    score += 1 * min(answer.count(term), 3)  # 최대 3번까지만
            
            if score > 0:
                results.append({
                    'metadata': metadata,
                    'score': score
                })
        
        # 점수 순으로 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"🔎 키워드 검색: {len(results)}개 결과 (상위 {k}개 사용)")
        
        # (metadata, score) 형태로 반환
        return [(r['metadata'], r['score']) for r in results[:k]]
    
    def _merge_results(
        self, 
        vector_results: List[tuple], 
        keyword_results: List[tuple]
    ) -> List[tuple]:
        """
        벡터 검색과 키워드 검색 결과 병합
        
        Args:
            vector_results: [(metadata, distance), ...]
            keyword_results: [(metadata, score), ...]
        
        Returns:
            [(metadata, combined_distance), ...]
        """
        merged = {}
        
        # 벡터 검색 결과 정규화 (거리 → 점수)
        if vector_results:
            max_distance = max(dist for _, dist in vector_results)
            min_distance = min(dist for _, dist in vector_results)
            distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
        
        for metadata, distance in vector_results:
            doc_id = metadata.get('id', id(metadata))
            # 정규화된 벡터 점수 (0~1)
            normalized_score = 1 - ((distance - min_distance) / distance_range) if distance_range > 0 else 1.0
            
            merged[doc_id] = {
                'metadata': metadata,
                'vector_score': normalized_score,
                'keyword_score': 0
            }
        
        # 키워드 검색 결과 정규화
        if keyword_results:
            max_keyword_score = max(score for _, score in keyword_results) if keyword_results else 1.0
        else:
            max_keyword_score = 1.0
        
        for metadata, score in keyword_results:
            doc_id = metadata.get('id', id(metadata))
            normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
            
            if doc_id in merged:
                merged[doc_id]['keyword_score'] = normalized_score
            else:
                merged[doc_id] = {
                    'metadata': metadata,
                    'vector_score': 0,
                    'keyword_score': normalized_score
                }
        
        # 최종 점수 계산 (벡터 60% + 키워드 40%)
        # 기술 용어가 많은 FAQ 특성상 키워드 비중 증가
        vector_weight = 0.6
        keyword_weight = 0.4
        
        for doc_id in merged:
            merged[doc_id]['final_score'] = (
                vector_weight * merged[doc_id]['vector_score'] +
                keyword_weight * merged[doc_id]['keyword_score']
            )
        
        # 점수 순으로 정렬
        sorted_results = sorted(
            merged.values(), 
            key=lambda x: x['final_score'], 
            reverse=True
        )
        
        logger.info(f"🔗 하이브리드 병합: {len(sorted_results)}개 결과")
        
        # (metadata, pseudo_distance) 형태로 반환 (기존 코드와 호환)
        # final_score가 높을수록 distance는 낮아야 함
        return [
            (item['metadata'], 1.0 / (1.0 + item['final_score']) if item['final_score'] > 0 else 10.0)
            for item in sorted_results
        ]
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[tuple], 
        top_k: int = 10
    ) -> List[tuple]:
        """
        Cross-Encoder로 검색 결과 재정렬
        
        Args:
            query: 검색 쿼리
            results: [(metadata, distance), ...]
            top_k: 재랭킹할 결과 개수
        
        Returns:
            재정렬된 [(metadata, distance), ...]
        """
        if not self.reranker or not results:
            return results
        
        try:
            # 재랭킹할 결과 선택 (상위 top_k*2개)
            candidates = results[:min(len(results), top_k * 2)]
            
            # (질문, FAQ 질문+답변 일부) 쌍 생성
            pairs = []
            for metadata, _ in candidates:
                question = metadata.get('question', '')
                answer = metadata.get('answer', '')[:300]  # 답변 일부만
                faq_text = f"{question} {answer}"
                pairs.append([query, faq_text])
            
            # Reranking 점수 계산
            scores = self.reranker.predict(pairs)
            
            logger.info(f"🎯 Reranking: {len(candidates)}개 후보 재정렬")
            
            # 점수 순으로 재정렬
            reranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # 재정렬된 결과 + 나머지 결과 결합
            reranked_results = [item[0] for item in reranked[:top_k]]
            remaining_results = results[len(candidates):]
            
            return reranked_results + remaining_results
            
        except Exception as e:
            logger.warning(f"⚠️ Reranking 실패, 원본 결과 사용: {e}")
            return results
    
    async def search_faq(
        self, 
        query: str, 
        k: int = 5,
        category_filter: Optional[str] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True
    ) -> List[Dict]:
        """
        FAQ 검색 (Hybrid + Reranking)
        
        Args:
            query: 사용자 질문
            k: 반환할 결과 개수
            category_filter: 대분류 필터
            use_hybrid: 하이브리드 검색 사용 여부
            use_reranking: 재랭킹 사용 여부
        
        Returns:
            검색 결과 리스트
        """
        try:
            logger.info(f"🔍 검색 쿼리: {query}")
            
            # 1. 벡터 검색
            query_embedding = await self.embedding_service.embed_text(query)
            search_k = k * 4 if category_filter else k * 3
            vector_results = self.faiss_service.search(query_embedding, k=search_k)
            logger.info(f"📊 벡터 검색: {len(vector_results)}개 결과")
            
            # 2. 하이브리드 검색
            if use_hybrid:
                keyword_results = self._keyword_search(query, k=search_k)
                raw_results = self._merge_results(vector_results, keyword_results)
            else:
                raw_results = vector_results
            
            # 3. 재랭킹
            if use_reranking and self.reranker:
                raw_results = self._rerank_results(query, raw_results, top_k=search_k)
            
            # 4. 카테고리 필터링
            filtered_results = []
            for metadata, distance in raw_results:
                if category_filter:
                    if metadata['category'] == category_filter:
                        filtered_results.append((metadata, distance))
                else:
                    filtered_results.append((metadata, distance))
            
            # 5. 상위 k개만 선택
            filtered_results = filtered_results[:k]
            
            # 6. 결과 포맷팅
            formatted_results = []
            for metadata, distance in filtered_results:
                similarity = self._calculate_similarity(distance)
                
                formatted_results.append({
                    "question": metadata['question'],
                    "answer": metadata['answer'],
                    "category": metadata['category'],
                    "domain": metadata['domain'],
                    "keywords": metadata['keywords'],
                    "cohort": metadata['cohort'],
                    "author": metadata['author'],
                    "distance": float(distance),
                    "similarity": float(similarity)
                })
                
                logger.debug(f"  - Distance: {distance:.3f} → Similarity: {similarity:.1%}")
            
            logger.info(f"✅ {len(formatted_results)}개 결과 검색 완료")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ FAQ 검색 실패: {e}")
            raise
    
    async def generate_answer(
        self, 
        query: str, 
        context_faqs: List[Dict],
        use_streaming: bool = False
    ) -> str:
        """RAG 기반 답변 생성"""
        try:
            # 1. 컨텍스트 구성
            context = self._build_context(context_faqs)
            
            # 2. 프롬프트 생성
            prompt = self._create_rag_prompt(query, context)
            
            # 3. GPT 답변 생성
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.3,
                stream=use_streaming
            )
            
            if use_streaming:
                return response
            else:
                answer = response.choices[0].message.content.strip()
                logger.info(f"✅ 답변 생성 완료 ({len(answer)}자)")
                return answer
            
        except Exception as e:
            logger.error(f"❌ 답변 생성 실패: {e}")
            raise
    
    def _build_context(self, faqs: List[Dict]) -> str:
        """검색된 FAQ들을 컨텍스트로 구성"""
        if not faqs:
            return "관련된 FAQ를 찾을 수 없습니다."
        
        context_parts = []
        for i, faq in enumerate(faqs, 1):
            # 키워드 안전 처리
            keywords = [k for k in faq.get('keywords', []) if k and k.strip()]
            keywords_str = ', '.join(keywords) if keywords else '없음'
            
            context_parts.append(
                f"### FAQ {i} (유사도: {faq['similarity']:.1%})\n"
                f"**카테고리**: {faq['category']}\n"
                f"**키워드**: {keywords_str}\n"
                f"**질문**: {faq['question']}\n"
                f"**답변**: {faq['answer']}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 SSAFY(삼성 청년 SW 아카데미) 코치 역할의 AI 어시스턴트입니다.

**역할**:
- 교육생들의 기술 질문에 정확하고 친절하게 답변
- 제공된 과거 FAQ 사례를 참고하여 답변
- 실무 경험을 바탕으로 한 구체적인 가이드 제공

**답변 원칙**:
1. 제공된 FAQ 컨텍스트를 우선적으로 활용
2. 단순 복사가 아닌, 질문자의 상황에 맞게 재구성
3. 구체적인 코드 예시나 해결 방법 제시 (코드는 ```로 감싸기)
4. 추가 학습 자료나 참고 링크 제안 가능
5. 확실하지 않은 내용은 추측하지 말고 명확히 표현

**답변 구조**:
핵심 답변: 문제의 핵심과 해결 방법을 2-3문장으로 요약

상세 설명:
1. 첫 번째 단계나 원인 설명
2. 두 번째 단계나 해결 방법
3. 세 번째 단계나 추가 정보

필요한 경우 코드 예시를 제공하세요:
```언어
코드 예시
```

추가 팁: 주의사항이나 참고할 만한 내용
"""
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """RAG 프롬프트 생성"""
        return f"""아래는 과거 유사한 질문에 대한 FAQ 답변 사례입니다.

{context}

---

**현재 질문**: {query}

위의 FAQ 사례들을 참고하여, 현재 질문에 대해 정확하고 도움이 되는 답변을 작성해주세요.
과거 사례와 유사하더라도 질문자의 구체적인 상황에 맞게 답변을 재구성하고, 
필요한 경우 추가적인 설명이나 예시를 제공해주세요.

**코드 예시를 포함할 때는 반드시 마크다운 코드 블록(```)을 사용하세요.**
"""
    
    async def query(
        self, 
        question: str, 
        k: int = 3,
        category_filter: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict:
        """
        통합 RAG 쿼리 (Hybrid + Reranking + 답변 생성)
        
        Args:
            question: 사용자 질문
            k: 검색할 FAQ 개수
            category_filter: 카테고리 필터
            return_sources: 출처 포함 여부
        
        Returns:
            {
                "question": "...",
                "answer": "...",
                "sources": [...],
                "metadata": {...}
            }
        """
        try:
            # 1. FAQ 검색 (Hybrid + Reranking)
            search_results = await self.search_faq(
                query=question,
                k=k,
                category_filter=category_filter,
                use_hybrid=True,      # ✅ 하이브리드 검색
                use_reranking=True    # ✅ 재랭킹
            )
            
            # 2. 답변 생성
            answer = await self.generate_answer(
                query=question,
                context_faqs=search_results
            )
            
            # 3. 결과 구성
            result = {
                "question": question,
                "answer": answer,
                "metadata": {
                    "num_sources": len(search_results),
                    "category_filter": category_filter,
                    "top_similarity": search_results[0]['similarity'] if search_results else 0,
                    "search_method": "hybrid + reranking"  # ✅ 검색 방법 표시
                }
            }
            
            if return_sources:
                result["sources"] = search_results
            
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG 쿼리 실패: {e}")
            raise