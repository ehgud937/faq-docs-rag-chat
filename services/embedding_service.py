from typing import List
from openai import AsyncOpenAI
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """OpenAI Embedding 생성"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.GMS_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        self.model = "text-embedding-3-small"  # 빠르고 저렴
        # self.model = "text-embedding-3-large"  # 더 정확
    
    async def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        대량 텍스트 임베딩 (배치 처리)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: API 호출당 처리 개수
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"임베딩 진행: {i+1}~{min(i+batch_size, len(texts))}/{len(texts)}")
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"배치 {i} 실패: {e}")
                raise
        
        logger.info(f"✅ 총 {len(embeddings)}개 임베딩 완료")
        return embeddings