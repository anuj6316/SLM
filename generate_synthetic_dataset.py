from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import Dataset
from bonito import Bonito, SamplingParams

from unstructured_bonito import Bonito, SamplingParams

def main():
    # Load the markdown file
    loader = UnstructuredMarkdownLoader(
        "/home/mindmap/Desktop/SLM/unstructured_data/cleaned_output.md",
        mode="single",
        strategy="fast",
    )
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    print(f"Generated {len(texts)} chunks from {len(documents)} documents.")

    # 1. Prepare your clean Markdown chunks in a dictionary format
    # Extract page_content from Document objects to ensure we pass strings to the dataset
    text_chunks = [doc.page_content for doc in texts]
    
    data = {"sentence": text_chunks}
    dataset = Dataset.from_dict(data)

    # 2. Initialize the Bonito model
    bonito = Bonito("BatsResearch/bonito-v1")

    # 3. Define generation parameters
    sampling_params = SamplingParams(
        max_tokens=256, 
        top_p=0.95, 
        temperature=0.5, 
        n=1
    )

    # 4. Generate the synthetic dataset
    synthetic_dataset = bonito.generate_tasks(
        dataset, 
        context_col="sentence", 
        task_type="qg", # qg = Question Generation
        sampling_params=sampling_params 
    )

    print(synthetic_dataset)

if __name__ == "__main__":
    main()