import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    qwen_api_key: str = os.getenv("DASHSCOPE_API_KEY", os.getenv("QWEN_API_KEY", ""))
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

settings = Settings()
