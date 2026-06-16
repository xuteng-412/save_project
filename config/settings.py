"""
Application settings loaded from environment variables and `.env`.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_API_BASE: str = "http://localhost:8000/v1"
    MODEL_NAME: str = "qwen2.5-7b-instruct"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # Database
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "psy_agent"
    USE_DATABASE: bool = True

    # Cache / app
    REDIS_URL: str = "redis://localhost:6379/0"
    DEBUG: bool = True
    MAX_HISTORY_TURNS: int = 10

    # 四模块并行开发：为 true 时使用 Mock，false 时使用 Stub（可替换为真实实现）
    MOCK_SAFETY: bool = True
    MOCK_EMOTION: bool = True
    MOCK_ROUTER: bool = True
    MOCK_INTERVENTION: bool = True

    # ASR（听写）：sensevoice 偏中文口语场景；无 funasr 或失败时自动回退 whisper
    ASR_BACKEND: str = "sensevoice"
    SENSEVOICE_ASR_MODEL: str = "iic/SenseVoiceSmall"
    SENSEVOICE_DEVICE: str = "cuda:0"

    # Faster-Whisper（ASR_BACKEND=whisper 或作为 sensevoice 的回退）
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_MODEL_PATH: str = (
        "models/models--Systran--faster-whisper-base/snapshots/"
        "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
    )
    WHISPER_DEVICE: str = "auto"

    # Multimodal emotion stack
    AUDIO_EMOTION_BACKEND: str = "sensevoice"
    VISUAL_EMOTION_BACKEND: str = "emotiefflib"
    ENABLE_MULTIMODAL_EMOTION_FUSION: bool = True

    # 安全过滤 — 多模态（ASR 后端由 ASR_BACKEND 统一控制）
    SAFETY_NSFW_MODEL_PATH: str = ""
    SAFETY_DEVICE: str = "cuda"
    SAFETY_MODEL_TYPE: str = "nsfw"  # "nsfw"（仅色情）或 "clip"（多类别零样本：色情/暴力/血腥/自残/毒品）

    # Emotion-LLaMA sidecar
    EMOTION_LLAMA_ENABLED: bool = False
    EMOTION_LLAMA_GRADIO_URL: str = "http://127.0.0.1:7889"
    EMOTION_LLAMA_TIMEOUT: float = 180.0

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
