from src.preprocessing import create_loader
def test_hf_loader():
    config = {
        "source_type": "hf",
        "path": "google/WaxalNLP",
        "name": "ach_asr",
	    "ignore_columns": ["id"],
        "split": "train",
        "limit": 10
    }
    
    loader = create_loader(config)
    loader.preview(3)
    
    for example in loader.stream_with_limit(5):
        print(example)
if __name__ == "__main__":
    test_hf_loader()
