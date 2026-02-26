"""
MLflow Traced Inference for Text-to-SQL with Qwen2.5-Coder-3B

Sends traces to Databricks MLflow Cloud for observability.

Usage:
    1. Copy .env.example to .env and fill in Databricks credentials
    2. Run: python src/tracing/inference_databricks.py

View traces in Databricks:
    AI/ML -> Experiments -> text2sql-inference -> Traces tab
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import mlflow

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/text2sql-inference"))

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-3B-Instruct")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
LOAD_4BIT = os.getenv("LOAD_4BIT", "true").lower() == "true"

_model = None
_tokenizer = None


def get_model_and_tokenizer():
    """Lazy load model and tokenizer (singleton pattern)."""
    global _model, _tokenizer

    if _model is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        try:
            from unsloth import FastLanguageModel

            _model, _tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=LOAD_4BIT,
                dtype=None,
            )
            FastLanguageModel.for_inference(_model)
            logger.info("Model loaded successfully")
        except ImportError:
            logger.warning("Unsloth not available, falling back to transformers")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype="auto",
            )
            logger.info("Model loaded with transformers (fallback)")

    return _model, _tokenizer


@mlflow.trace(
    name="generate_sql",
    span_type="LLM",
)
def generate_sql(prompt: str, max_tokens: int = 128) -> Dict[str, Any]:
    """
    Generate SQL from a formatted prompt.

    Args:
        prompt: The full prompt including schema and question
        max_tokens: Maximum tokens to generate

    Returns:
        Dict with sql, input_tokens, output_tokens
    """
    import torch

    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    sql = clean_sql_output(generated_text)

    mlflow.log_metrics(
        {
            "input_tokens": input_length,
            "output_tokens": len(new_tokens),
            "total_tokens": input_length + len(new_tokens),
        }
    )

    return {
        "sql": sql,
        "raw_output": generated_text,
        "input_tokens": input_length,
        "output_tokens": len(new_tokens),
    }


def clean_sql_output(text: str) -> str:
    """Clean the generated SQL output."""
    text = text.replace("```sql", "").replace("```", "").strip()
    text = " ".join(text.split())

    end_markers = ["<|im_end|>", "<|endoftext|>", "###", "\n\n"]
    for marker in end_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    return text


@mlflow.trace(name="load_schema", span_type="RETRIEVER")
def load_schema(db_id: str, schema_path: Optional[str] = None) -> str:
    """
    Load database schema by db_id.

    Args:
        db_id: Database identifier
        schema_path: Path to schema file (tables.json format)

    Returns:
        Schema string in format: "table1(col1, col2) | table2(col3)"
    """
    default_schemas = {
        "concert_singer": "singer(singer_id, name, country, age) | stadium(stadium_id, name, capacity)",
        "employee": "employee(id, name, dept_id, salary) | department(dept_id, name, location)",
        "student_course": "student(id, name, age) | course(id, name, credits) | enrollment(student_id, course_id, grade)",
    }

    if schema_path and os.path.exists(schema_path):
        try:
            with open(schema_path, "r") as f:
                tables_data = json.load(f)
            for db in tables_data:
                if db.get("db_id") == db_id:
                    return format_schema_from_tables_json(db)
        except Exception as e:
            logger.warning(f"Failed to load schema from {schema_path}: {e}")

    return default_schemas.get(db_id, f"Schema for {db_id} not found")


def format_schema_from_tables_json(db_info: Dict[str, Any]) -> str:
    """Format schema from tables.json format to string."""
    table_names = db_info.get("table_names_original", [])
    column_names = db_info.get("column_names_original", [])

    tables_dict: Dict[str, list] = {}
    for table_idx, col_name in column_names:
        if table_idx == -1:
            continue
        t_name = table_names[table_idx]
        tables_dict.setdefault(t_name, []).append(col_name)

    parts = [f"{t}({', '.join(cols)})" for t, cols in tables_dict.items()]
    return " | ".join(parts)


@mlflow.trace(name="build_prompt", span_type="CHAIN")
def build_prompt(
    schema: str, question: str, system_prompt: Optional[str] = None
) -> str:
    """
    Build the full prompt for the model.

    Args:
        schema: Database schema string
        question: Natural language question
        system_prompt: Optional system prompt override

    Returns:
        Formatted prompt string
    """
    if system_prompt is None:
        system_prompt = "You are an expert Text-to-SQL assistant. Convert the natural language question into a valid SQL query based on the schema."

    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
### Database Schema:
{schema}

### Question:
{question}<|im_end|>
<|im_start|>assistant
"""
    return prompt


@mlflow.trace(name="text_to_sql_pipeline", span_type="CHAIN")
def text_to_sql_pipeline(
    question: str,
    db_id: str = "default",
    schema_path: Optional[str] = None,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Full Text-to-SQL pipeline with tracing.

    Args:
        question: Natural language question
        db_id: Database identifier
        schema_path: Path to schema file
        max_tokens: Maximum tokens to generate

    Returns:
        Dict with question, db_id, sql, tokens_used
    """
    schema = load_schema(db_id, schema_path)
    prompt = build_prompt(schema, question)
    result = generate_sql(prompt, max_tokens)

    return {
        "question": question,
        "db_id": db_id,
        "schema": schema,
        "sql": result["sql"],
        "tokens_used": result["input_tokens"] + result["output_tokens"],
    }


def run_demo():
    """Run a demo with sample questions."""
    test_cases = [
        {
            "question": "How many singers are from USA?",
            "db_id": "concert_singer",
        },
        {
            "question": "What is the average salary of employees in the Engineering department?",
            "db_id": "employee",
        },
        {
            "question": "List all students enrolled in the Database course",
            "db_id": "student_course",
        },
    ]

    with mlflow.start_run(run_name="inference-demo"):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "max_seq_length": MAX_SEQ_LENGTH,
                "load_4bit": LOAD_4BIT,
            }
        )

        for i, tc in enumerate(test_cases):
            logger.info(f"\n--- Test Case {i + 1} ---")
            logger.info(f"Question: {tc['question']}")
            logger.info(f"DB: {tc['db_id']}")

            result = text_to_sql_pipeline(
                question=tc["question"],
                db_id=tc["db_id"],
            )

            logger.info(f"Generated SQL: {result['sql']}")
            logger.info(f"Tokens used: {result['tokens_used']}")

            print(f"\nQ: {tc['question']}")
            print(f"SQL: {result['sql']}")

def run_inference(data_path: str, num_samples: Optional[int], output_path: Optional[str]):
    """Run inference on dataset with tracing."""
    data = load_dataset(data_path, num_samples)
    results = []
    
    with mlflow.start_run(run_name="val-inference"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "dataset": data_path,
            "num_samples": len(data),
        })
        
        for item in tqdm(data, desc="Generating SQL"):
            result = text_to_sql_pipeline(
                question=item["question"],
                db_id=item["db_id"],
            )
            result["gold_sql"] = item["gold_sql"]
            result["exact_match"] = result["sql"].lower() == item["gold_sql"].lower()
            results.append(result)
        
        # Log aggregate metrics
        exact_matches = sum(1 for r in results if r["exact_match"])
        mlflow.log_metric("exact_match_accuracy", exact_matches / len(results))
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    return results

def load_dataset(data_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            entry = json.loads(line)
            data.append({
                "question": extract_question(entry),
                "db_id": entry.get("metadata", {}).get("db_id", "unknown"),
                "gold_sql": entry.get("output", ""),
            })
    return data

def extract_question(entry: Dict) -> str:
    """Extract question from entry (handles different formats)."""
    if "input" in entry:
        # Format: "### Database Schema:...\n### Question:\n{question}"
        input_text = entry["input"]
        if "### Question:" in input_text:
            return input_text.split("### Question:")[-1].strip()
    return entry.get("question", "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Traced Text-to-SQL Inference")
    parser.add_argument("--data_path", type=str, default="data/val_split.jsonl",
                        help="Path to JSONL dataset")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (None = all)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()
