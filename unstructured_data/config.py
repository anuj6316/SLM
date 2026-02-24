from dataclasses import dataclass
from typing import List

@dataclass
class scrapeConfig:
    url: str
    api_key: str
    active_url: str = None
    content_path: str = None
    status_code: int = None
    links: list = None

@dataclass
class ProcessMarkdownQAPairsConfig:
    api_key: str
    model_id: str
    

@dataclass 
class qaPairs:
    question: str
    answer: str
    judge_review: str

@dataclass 
class jsonlFormat:
    chunk_content: str
    qa_pairs: List[qaPairs]
