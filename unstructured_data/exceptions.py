"""
Custom exceptions for the unstructured Data pipeline
"""
## base Class exception for this pipeline
class PipelineError(Exception):
    pass
## 1. configuration error: missing fields inside config.yml
class ConfigurationError(PipelineError):
    pass

## 2. ScarpeError: 
class ScrapeError(PipelineError):
    def __init__(self, message: str, url=None, status_code=None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code

## 3. LLMGenerationError: when LLM fails to follow the pydantic schema
class LLMResponseError(PipelineError):
    pass

## 4. ProcessingError
class ProcessingError(PipelineError):
    pass

## 5. RateLimitError
class RateLimitError(PipelineError):
    pass