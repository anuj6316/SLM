from dotenv import load_dotenv
import os
import yaml
from pydantic import BaseModel
import logging 
from pprint import pprint
from openai import AsyncOpenAI
import asyncio
import json
import random

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        raw_yaml = f.read()
        expanded_yaml = os.path.expandvars(raw_yaml)
        cfg = yaml.safe_load(expanded_yaml)
    return cfg

async def call_llm(cfg: LLMConfig, prompt: str, role: str, system_prompt: str = ""):
    client = AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url
    )
    response = await client.chat.completions.create(
        model=cfg.model,
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=cfg.temperature,
        response_format = {"type": "json_object"}
    )
    return response.choices[0].message.content

def get_random_seeds(file_path: str, n=3):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    samples = [json.loads(line) for line in random.sample(lines, n)]
    return samples

def main():
    # 1. API keys
    cfg = load_config("/home/mindmap/Desktop/SLM/scripts/config.yml")
    pprint(cfg)
    # judge = LLMConfig(provider = "judge", model = cfg['judge']['model'], api_key = cfg['judge']['api_key'], base_url = cfg['judge']['base_url'], temperature = cfg['judge']['temperature'])
    # provider = LLMConfig(provider = "generator", model = cfg['generator']['model'], api_key = cfg['generator']['api_key'], base_url = cfg['generator']['base_url'], temperature = cfg['generator']['temperature'])
    judge_config = LLMConfig(**cfg['judge'])
    generator_config = LLMConfig(**cfg['generator'])
    print("\n--- Access Individual Fields ---")
    print("Provider:", judge_config.provider)
    print("Model:", judge_config.model)
    print("API Key:", judge_config.api_key)
    print("Base URL:", judge_config.base_url)
    print("Temperature:", judge_config.temperature)  

    ## getting random instruction field for dataset generation
    instruction = get_random_seeds("/home/mindmap/Desktop/SLM/data/spider_sft/train_sft.jsonl")
    pprint(instruction)  
if __name__ == "__main__":
    main()