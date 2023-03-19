from pydantic import BaseSettings


class VoxioRuntimeSettings(BaseSettings):
    index_regex: str = r"\d+"


settings = VoxioRuntimeSettings()
