from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from distilabel.llms.openai import OpenAILLM

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


SYSTEM_PROMPT = """
You are an expert assistant.
Generate high-quality question and answer pairs based on the provided document.
Return them clearly formatted.
"""


def load_and_chunk_markdown(file_path: str):
    loader = UnstructuredMarkdownLoader(
        file_path,
        mode="single",
        strategy="fast",
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")

    return [{"page": chunk.page_content} for chunk in chunks]


if __name__ == "__main__":
    data = load_and_chunk_markdown(
        "/home/mindmap/Desktop/SLM/unstructured_data/cleaned_output.md"
    )

    with Pipeline(name="qa_generation_openai") as pipeline:

        load_dataset = LoadDataFromDicts(
            name="load_dataset",
            data=data,
            batch_size=4,
        )

        text_generation = TextGeneration(
            name="qa_generation",
            system_prompt=SYSTEM_PROMPT,
            template="""
Generate 5 question-answer pairs about the following document:

Document:
{{ page }}
""",
            llm=OpenAILLM(
                model="gpt-4o-mini",   # Fast + cheap
            ),
            input_batch_size=4,
        )

        load_dataset >> text_generation

    distiset = pipeline.run(
        parameters={
            "qa_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_tokens": 800,
                        "temperature": 0.7,
                    }
                }
            }
        },
        use_cache=False,
    )

    print(distiset)