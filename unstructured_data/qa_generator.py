from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from prompts.question_generation import q_prompt, a_prompt, judge_prompt
from prompts.system_prompt import q_system_prompt, a_system_prompt, judge_system_prompt
from pprint import pprint
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, RootModel
from typing import List
from datetime import datetime
from dataclasses import asdict
from groq import RateLimitError
import json
import re

## Config
from config import QAPairs, JsonlFormat, ProcessMarkdownQAPairsConfig
from utils import (
    chunk_hash,
    text_splitter,
    load_config
)

## Errors
from exceptions import (
    PipelineError,
    ConfigurationError,
    ScrapeError,
    LLMResponseError,
    ProcessingError,
    RateLimitError
)
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# pydantic output parser
class Question(BaseModel):
    question: str

class QuestionOutput(RootModel):
    root: List[Question]

class AnswerOutput(BaseModel):
    answer: str

class JudgeOutput(BaseModel):
    score: float
    reasoning: str

class ProcessMarkdownQAPairs:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = load_config(self.config_path)['ProcessMarkdownQAPairs']
        self.llm = self.initialize_model(self.cfg.model_id, self.cfg.api_key)

    def initialize_model(self, model_id: str, api_key: str) -> ChatGroq:
        """Initializes the model for future use

        Args:
            model_id (str): _description_
            api_key (str): _description_

        Returns:
            ChatGroq: Langchain chatgroq object.
        """
        llm = ChatGroq(
                model=model_id,
                temperature=0,
                max_tokens=None,
                # reasoning_format="parsed",
                timeout=None,
                max_retries=2,
                api_key=os.getenv("GROQ_API_KEY"),
                # other params...
            )        
        return llm

    def safe_invoke(self, messages:  List[dict], max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                cleaned_response = self.cleaned_response(messages)
                return cleaned_response
                # return self.llm.invoke(messages) 
            except RateLimitError:
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
        raise Exception("Max retries exceeded.")

    def cleaned_response(self, messages: List[dict]) -> str:
        response = self.llm.invoke(messages)
        content = response.content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        print(f"\n\n{content}\n\n")
        return content.strip('\n')

    def generate_questions(self, chunk: str) -> QuestionOutput:
        ## 1. Initialize Pydantic output parser
        question_parser = PydanticOutputParser(pydantic_object=QuestionOutput)

        ## 2. Prompt template
        prompt_template = PromptTemplate(
            input_variables=["chunk", "schema"],
            template=q_prompt,
        )

        ## 3. PT -> Rendered Prompt
        rendered_output = prompt_template.format(
            chunk=chunk,
            schema=question_parser.get_format_instructions()
        )

        ## 4. Message setup
        messages = [
            {'role': 'system', "content": q_system_prompt},
            {'role': 'user', "content": rendered_output}
        ]

        ## 5. generating response
        # ai_response = self.llm.invoke(messages)
        cleaned_output = self.cleaned_response(messages)
        try:
            ai_msg = question_parser.parse(cleaned_output)
            return ai_msg
        except Exception as e:
            print(f"PARSING FAILED. Cleaned Output: {cleaned_output}")
            raise e

        return

    def generate_answers(self, chunk: str, questions: QuestionOutput) -> AnswerOutput:
        ## 1. Initialize Pydantic output parser
        answer_parser = PydanticOutputParser(pydantic_object=AnswerOutput)

        ## 2. Prompt template
        prompt_template = PromptTemplate(
            input_variables = ["schema", "chunk", "question"],
            template = a_prompt,
        )

        ## 3. PT -> Rendered Prompt
        ## 4. Message setup for multiple questions
        messages = []
        for question in questions.root:
            rendered_output = prompt_template.format(
                schema=answer_parser.get_format_instructions(),
                chunk=chunk,
                question=question.question
            )
            messages.append([
                {'role': 'system', 'content': a_system_prompt},
                {'role': 'user', 'content': rendered_output}
            ])

        ## 5. generating answer form questions using batch
        # ai_response = self.llm.batch(messages) ## TPM error
        ai_response = [self.safe_invoke(msg) for msg in messages]

        ## 6. Parse each response
        try:
            ai_msgs = [answer_parser.parse(resp) for resp in ai_response]
            ## 7. return
            return ai_msgs
        except Exception as e:
            print(f"RAW OUTPUT: {ai_response.content}")
            raise e

        raise LLMResponseError

    def generate_qa_pairs(self, chunk: str, questions: QuestionOutput, answers: AnswerOutput) -> List[QAPairs]:
        qa_pairs = []

        for question, answer in zip(questions.root, answers):
            judge_response = self.judge_qa_pair(chunk, question.question, answer.answer)

            qa_pair = QAPairs(
                question=question.question,
                answer=answer.answer,
                judge_review=judge_response.reasoning,
                judge_score=judge_response.score
            )
            qa_pairs.append(qa_pair)

        return qa_pairs

    def judge_qa_pair(self, chunk: str, question: str, answer: str) -> JudgeOutput:
        ## 1. Initialize Pydantic output parser
        judge_parser = PydanticOutputParser(pydantic_object=JudgeOutput)

        ## 2. Prompt Template
        prompt_template = PromptTemplate(
            input_variables=["schema", "chunk", "question", "answer"],
            template=judge_prompt
        )

        ## 3. PT -> Rendered Prompt
        rendered_output = prompt_template.format(
            schema=judge_parser.get_format_instructions(),
            chunk=chunk,
            question=question,
            answer=answer
        )
        
        ## 4. Message setup
        messages = [
            {'role': 'system', 'content': judge_system_prompt},
            {'role': 'user', 'content': rendered_output}
        ]

        ## 5. generating response
        ai_response = self.cleaned_response(messages)
        ai_msg = judge_parser.parse(ai_response)
        # print(ai_msg)

        ## 6. return 
        return ai_msg

    def append_to_jsonl(self, record: JsonlFormat):
        with open("final_output.jsonl", 'a', encoding='utf-8') as f:
            ## 1. convert dataclass to dict
            json_string = json.dumps(asdict(record), ensure_ascii=False)
            f.write(json_string + "\n")

    def run(self):
        with open(self.cfg.file_path, "r") as f:
            data = f.read()
        chunks = text_splitter(data)
        for idx, chunk in enumerate(chunks):
            try:
                questions = self.generate_questions(chunk)
                answers = self.generate_answers(chunk, questions)
                qa_pairs = self.generate_qa_pairs(chunk, questions, answers)
                # print(qa_pairs)

                try:
                    record = JsonlFormat(
                        chunk_id = chunk_hash(chunk),
                        chunk_content = chunk,
                        qa_pairs = qa_pairs,
                        metadata= {
                            "file_path": self.cfg.file_path,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "chunk_index": idx,
                            "chunk_length": len(chunk)
                        }
                    )
                    self.append_to_jsonl(record)
                except Exception as e:
                    logging.error(e)
            except Exception as e:
                logging.error(f"Chunk {idx} failed: {e}")
        # pass

if __name__ == "__main__":
    obj = ProcessMarkdownQAPairs()
    obj.run()