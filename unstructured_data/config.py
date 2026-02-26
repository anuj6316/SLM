from dataclasses import dataclass
from typing import List, Dict, Any, Literal


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

@dataclass
class ProcessMarkdownQAPairsConfig:
    api_key: str
    model_id: str
    file_path: str

@dataclass 
class QAPairs:
    question: str
    answer: str
    judge_review: str
    judge_score: float

@dataclass 
class JsonlFormat:
    chunk_id: str
    chunk_content: str
    qa_pairs: List[QAPairs]
    metadata: Dict[str, Any]
