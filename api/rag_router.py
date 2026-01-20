from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from services.rag_service import RAGService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG"])

# 전역 RAG 서비스 인스턴스
rag_service = RAGService()


# === Request/Response 모델 ===

class SearchRequest(BaseModel):
    query: str = Field(..., description="검색 질문", min_length=1)
    k: int = Field(5, description="검색 결과 개수", ge=1, le=20)
    category: Optional[str] = Field(None, description="카테고리 필터 (BE/FE/DB/Infra 등)")


class QueryRequest(BaseModel):
    question: str = Field(..., description="사용자 질문", min_length=1)
    k: int = Field(3, description="참고할 FAQ 개수", ge=1, le=10)
    search_k: int = Field(10, description="검색할 전체 FAQ 개수", ge=1, le=30)  
    category: Optional[str] = Field(None, description="카테고리 필터")
    return_sources: bool = Field(True, description="출처 포함 여부")
    return_all_candidates: bool = Field(True, description="모든 후보 FAQ 포함 여부")


class FAQResult(BaseModel):
    question: str
    answer: str
    category: str
    domain: str
    keywords: List[str]
    cohort: str
    author: str
    distance: float
    similarity: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[FAQResult]] = None
    all_candidates: Optional[List[FAQResult]] = None
    metadata: Dict


# === API 엔드포인트 ===

@router.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "FAQ RAG",
        "index_loaded": rag_service.faiss_service.index is not None
    }


@router.post("/search", response_model=List[FAQResult])
async def search_faq(request: SearchRequest):
    """
    FAQ 검색만 수행 (답변 생성 없음)
    
    - **query**: 검색할 질문
    - **k**: 반환할 결과 개수 (기본 5개)
    - **category**: 카테고리 필터 (선택)
    """
    try:
        results = await rag_service.search_faq(
            query=request.query,
            k=request.k,
            category_filter=request.category
        )
        
        return results
        
    except Exception as e:
        logger.error(f"검색 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """
    RAG 기반 질의응답 (검색 + 답변 생성)
    """
    try:
        result = await rag_service.query(
            question=request.question,
            k=request.k,
            search_k=request.search_k,
            category_filter=request.category,
            return_sources=request.return_sources,
            return_all_candidates=request.return_all_candidates
        )
        
        return result
        
    except Exception as e:
        logger.error(f"RAG 쿼리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories():
    """사용 가능한 카테고리 목록 조회"""
    if not rag_service.faiss_service.metadata:
        raise HTTPException(status_code=404, detail="인덱스가 로드되지 않았습니다")
    
    categories = set()
    for meta in rag_service.faiss_service.metadata:
        if meta.get('category'):
            categories.add(meta['category'])
    
    return {
        "categories": sorted(list(categories)),
        "count": len(categories)
    }


@router.get("/stats")
async def get_stats():
    """FAQ 인덱스 통계"""
    if not rag_service.faiss_service.metadata:
        raise HTTPException(status_code=404, detail="인덱스가 로드되지 않았습니다")
    
    total = len(rag_service.faiss_service.metadata)
    
    # 카테고리별 집계
    category_stats = {}
    domain_stats = {}
    
    for meta in rag_service.faiss_service.metadata:
        cat = meta.get('category', '미분류')
        dom = meta.get('domain', '무관')
        
        category_stats[cat] = category_stats.get(cat, 0) + 1
        domain_stats[dom] = domain_stats.get(dom, 0) + 1
    
    return {
        "total_faqs": total,
        "categories": category_stats,
        "domains": domain_stats,
        "index_dimension": rag_service.faiss_service.dimension
    }