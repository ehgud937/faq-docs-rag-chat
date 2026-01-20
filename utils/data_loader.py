# utils/data_loader.py

import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class FAQDataLoader:
    """CSV FAQ 데이터를 로드하고 전처리"""
    
    @staticmethod
    def load_faq_csv(file_path: str, encoding: str = 'utf-8') -> List[Dict]:
        """
        CSV 파일을 읽어 Document 형태로 변환
        
        Args:
            file_path: CSV 파일 경로
            encoding: 파일 인코딩 (기본: utf-8, 한글 깨지면 cp949 시도)
        
        Returns:
            [
                {
                    "content": "질문: ... 답변: ...",
                    "metadata": {...}
                },
                ...
            ]
        """
        try:
            # CSV 읽기 (인코딩 자동 감지)
            try:
                df = pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 디코딩 실패, cp949로 재시도...")
                df = pd.read_csv(file_path, encoding='cp949')
            
            logger.info(f"📊 CSV 로드: {len(df)}행 × {len(df.columns)}열")
            logger.info(f"📋 컬럼: {list(df.columns)}")
            
            # 필수 컬럼 확인
            required_columns = ['질문', '답변', '도메인', '대분류', '키워드']
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"필수 컬럼 누락: {missing}")
            
            documents = []
            skipped = 0
            
            for idx, row in df.iterrows():
                # None/NaN 값 처리
                question = str(row['질문']).strip() if pd.notna(row['질문']) else ""
                answer = str(row['답변']).strip() if pd.notna(row['답변']) else ""
                
                # 빈 값 스킵
                if not question or not answer or question == 'nan' or answer == 'nan':
                    logger.warning(f"Row {idx}: 질문 또는 답변이 비어있음 - 스킵")
                    skipped += 1
                    continue
                
                # 정확도를 위해 질문 + 답변 일부를 임베딩
                # 답변이 너무 길면 300자까지만
                content = f"질문: {question}\n\n답변: {answer[:300]}"
                
                # 키워드 파싱 (쉼표 구분)
                keywords_raw = str(row['키워드']) if pd.notna(row['키워드']) else ""
                keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
                
                metadata = {
                    "id": idx,
                    "domain": str(row['도메인']) if pd.notna(row['도메인']) else "무관",
                    "category": str(row['대분류']) if pd.notna(row['대분류']) else "",
                    "keywords": keywords,
                    "question": question,
                    "answer": answer,
                    "cohort": str(row['기수']) if pd.notna(row['기수']) and '기수' in row else "",
                    "author": str(row['작성자']) if pd.notna(row['작성자']) and '작성자' in row else ""
                }
                
                documents.append({
                    "content": content,
                    "metadata": metadata
                })
            
            logger.info(f"✅ {len(documents)}개 FAQ 로드 완료 (스킵: {skipped}개)")
            
            # 카테고리별 통계
            categories = {}
            for doc in documents:
                cat = doc['metadata']['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            logger.info(f"📊 카테고리별 분포: {categories}")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ CSV 로드 실패: {e}")
            raise