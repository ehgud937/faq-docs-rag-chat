# config/settings.py

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ========================================
    # GMS (통합 API) 설정
    # ========================================
    GMS_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"  # 기본값
    
    # ========================================
    # OpenAI 설정 (직접 사용하는 경우)
    # ========================================
    OPENAI_API_KEY: Optional[str] = None  # GMS 사용 시 불필요
    
    # ========================================
    # 임베딩 모델 설정
    # ========================================
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    # text-embedding-3-small: 빠르고 저렴 (1536차원)
    # text-embedding-3-large: 더 정확 (3072차원, 비쌈)
    EMBEDDING_DIMENSION: int = 1536
    
    # ========================================
    # GPT 모델 설정
    # ========================================
    GPT_MODEL: str = "gpt-4o-mini"
    # gpt-4o-mini: 빠르고 저렴
    # gpt-4o: 더 정확하지만 비쌈
    GPT_MAX_TOKENS: int = 1000
    GPT_TEMPERATURE: float = 0.3
    
    # ========================================
    # FAISS 인덱스 경로
    # ========================================
    FAISS_INDEX_PATH: str = "data/faiss_index"
    
    # ========================================
    # FAQ 데이터 경로
    # ========================================
    FAQ_CSV_PATH: str = "data/faq.csv"
    
    # ========================================
    # RAG 검색 설정
    # ========================================
    DEFAULT_SEARCH_K: int = 5  # 기본 검색 결과 개수
    DEFAULT_CONTEXT_K: int = 3  # GPT에 전달할 FAQ 개수
    MAX_SEARCH_K: int = 20
    
    # ========================================
    # MySQL 설정 (필요한 경우)
    # ========================================
    MYSQL_HOST: Optional[str] = None
    MYSQL_PORT: Optional[int] = None
    MYSQL_DATABASE: Optional[str] = None
    MYSQL_USERNAME: Optional[str] = None
    MYSQL_PASSWORD: Optional[str] = None
    
    # ========================================
    # MongoDB 설정 (필요한 경우)
    # ========================================
    MONGODB_HOST: Optional[str] = None
    MONGODB_PORT: Optional[int] = None
    MONGODB_DATABASE: Optional[str] = None
    MONGODB_USERNAME: Optional[str] = None
    MONGODB_PASSWORD: Optional[str] = None
    MONGODB_AUTH_DATABASE: Optional[str] = None
    
    # ========================================
    # JWT 설정 (필요한 경우)
    # ========================================
    JWT_SECRET: Optional[str] = None
    
    # ========================================
    # 로깅 설정
    # ========================================
    LOG_LEVEL: str = "INFO"
    
    # ========================================
    # 환경 설정
    # ========================================
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # 환경변수 대소문자 구분 안함


settings = Settings()