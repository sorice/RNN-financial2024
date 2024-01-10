from functools import lru_cache
from typing import Optional

from pydantic import AnyUrl

from pydantic_settings import BaseSettings

class BaseConfig(BaseSettings):
    testing: bool = False
    aws_region: str = "us-east-1"
    debug_level: str = "INFO"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    class Config:
        env_file = ".env"
        # case_sensitive = True  # FIXME: what is the reason this is case sensitive

@lru_cache
def get_config() -> BaseConfig:
    return BaseConfig()