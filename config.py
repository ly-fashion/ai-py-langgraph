"""项目配置模块"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """应用配置"""

    # OpenAI 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))

    # Agent 配置
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", "50"))

    @classmethod
    def validate(cls) -> None:
        """验证必要配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请在 .env 文件中配置，或设置环境变量。"
            )


config = Config()
