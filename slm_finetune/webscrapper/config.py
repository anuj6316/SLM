from dataclasses import dataclass
from typing import List, Dict, Any, Literal
from pydantic import BaseModel

@dataclass
class ScrapeConfig:
    url: str
    api_key: str
    scrape_type: Literal["flash", "deep"]
    active_url: str = None
    content_path: str = None
    status_code: int = None
    links: list = None
    remove_temp_files: bool = True

## Fastapi req and res
class HealthResponse(BaseModel):
    groq_isActive: bool
    jina_isActive: bool

class HealthRequest(BaseModel):
    groq_api_key: str
    jina_api_key: str

class ScrapeRequest(BaseModel):
    url: str
    api_key: str
    scrape_type: Literal["flash", "deep"]
    max_depth: int = 2
    max_workers: int = 5

class ScrapeResponse(BaseModel):
    job_id: str
    status: Literal["active", "completed", "failed"]
    number_of_links_scrapped: int
    created_at: str
    message: str