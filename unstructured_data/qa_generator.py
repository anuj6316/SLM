from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from prompts.question_generation import q_prompt, a_prompt
from prompts.system_prompt import q_system_prompt, a_system_prompt
from pprint import pprint
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, RootModel
from typing import List

## Config
from config import qaPairs, jsonlFormat, ProcessMarkdownQAPairsConfig
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

# class AnswerOutput(BaseModel):
#     __root__: List[Answer]

def text_splitter(document):
    """Creating multiple chunks form the raw text"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(document)
    return texts

def question_generation(chunk: str):
    """
    Goal:
        Generating High questions out of our chunks.
    args:
        chunk: str
    return:
        list of questions
    """
    # fixed pydantic output from our llm
    question_parser = PydanticOutputParser(pydantic_object=QuestionOutput)

    # prompt template
    prompt = PromptTemplate(
        input_variables=["chunk", "schema"],
        template=q_prompt,
    )

    # rendered prompt: text format
    rendered_prompt = prompt.format(chunk=chunk, schema=question_parser.get_format_instructions())
    print(rendered_prompt)

    # Intializing the LLM 
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
        # other params...
    )
    # Message setup
    messages = [
        {"role": "system", "content": q_system_prompt},
        {"role": "user", "content": rendered_prompt}
    ]
    # generating reponse
    response = llm.invoke(messages)
    # pprint(response.content)
    # for i in response.content:
    #     print(i)
    ai_msg = question_parser.parse(response.content)
    # pprint(type(ai_msg))
    for i in ai_msg.root:
        print(i.question)
    return ai_msg

def answer_generation(chunk: str, question: QuestionOutput):
    pass



class ProcessMarkdownQAPairs:

    def __init__(self, config_path: str="/home/mindmap/Desktop/SLM/unstructured_data/config.yml"):
        self.config_path = config_path
        self.cfg = self.load_config()
        self.llm = self.initialize_model(self.cfg.model_id, self.cfg.api_key)

        pass

    def load_config(self) -> dict:
        """Load the config.yml into a dataclass

        Returns:
            ProcessMarkdownQAPairsConfig: dataclass
        """
        with open(self.config_path, "r") as f:
            content = os.path.expandvars(f.read())  # <-- expands ${VAR}
            cfg = yaml.safe_load(content)
        
        cfg = ProcessMarkdownQAPairsConfig(**cfg['ProcessMarkdownQAPairs'])
        return cfg

    def initialize_model(self, model_id: str, api_key: str):
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
                reasoning_format="parsed",
                timeout=None,
                max_retries=2,
                api_key=os.getenv("GROQ_API_KEY"),
                # other params...
            )        
        return llm

    def call_llm(self):
        pass

    def generate_questions(self, chunk: str):
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
        ai_response = self.llm.invoke(messages)
        ai_msg = question_parser.parse(ai_response.content)
        return ai_msg

        ## 6. return list of questions
        pass

    def generate_answers(self, chunk: str, questions: QuestionOutput):
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
                question=question
            )
            messages.append(
                {'role': 'system', 'content': a_system_prompt},
                {'role': 'user', 'content': rendered_output}
            )

        ## 5. generating answer form questions using batch
        ai_response = self.llm.batch(messages)

        ## 6. Parse each response
        ai_msgs = [question_parser.parse(resp.content) for resp in ai_responses]

        ## 7. return
        return ai_msgs



    def judge_qa_pair(self):
        pass

    def save_file(self):
        pass

if __name__ == "__main__":
    obj = ProcessMarkdownQAPairs()
    obj.generate_questions()