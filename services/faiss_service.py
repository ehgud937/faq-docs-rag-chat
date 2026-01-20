import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class FAISSService:
    """FAISS 인덱스 관리 및 검색"""
    
    def __init__(self, index_path: str = "data/faiss_index"):
        self.index_path = index_path
        self.index = None
        self.metadata = []
        self.dimension = 1536  # text-embedding-3-small 차원
        
        os.makedirs(index_path, exist_ok=True)
    
    def build_index(self, embeddings: List[List[float]], metadata: List[Dict]):
        """
        FAISS 인덱스 생성
        
        Args:
            embeddings: 임베딩 벡터 리스트
            metadata: 각 임베딩에 대응하는 메타데이터
        """
        try:
            # NumPy 배열로 변환
            embedding_array = np.array(embeddings).astype('float32')
            
            # FAISS 인덱스 생성 (L2 거리 기반)
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # 정규화 추가 (cosine similarity 근사)
            # faiss.normalize_L2(embedding_array)
            
            # 인덱스에 벡터 추가
            self.index.add(embedding_array)
            self.metadata = metadata
            
            logger.info(f"✅ FAISS 인덱스 생성 완료: {self.index.ntotal}개 벡터")
            
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            raise
    
    def save_index(self):
        """인덱스 저장"""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            metadata_file = os.path.join(self.index_path, "metadata.json")
            
            faiss.write_index(self.index, index_file)
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 인덱스 저장 완료: {self.index_path}")
            
        except Exception as e:
            logger.error(f"인덱스 저장 실패: {e}")
            raise
    
    def load_index(self):
        """저장된 인덱스 로드"""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            metadata_file = os.path.join(self.index_path, "metadata.json")
            
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"인덱스 파일 없음: {index_file}")
            
            self.index = faiss.read_index(index_file)
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"✅ 인덱스 로드 완료: {self.index.ntotal}개 벡터")
            
        except Exception as e:
            logger.error(f"인덱스 로드 실패: {e}")
            raise
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        유사도 검색
        
        Args:
            query_embedding: 질문 임베딩 벡터
            k: 반환할 결과 개수
        
        Returns:
            [(metadata, distance), ...]
        """
        try:
            query_array = np.array([query_embedding]).astype('float32')
            
            # 검색 수행
            distances, indices = self.index.search(query_array, k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    results.append((self.metadata[idx], float(distance)))
            
            return results
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            raise