"""
=== Guide for gemma3 270M from scratch ===

    1. Loading Dataset.
    2. Dataset tokenization.
    3. Creating Input/Output pairs.
    4. Assembling Gemma3 270 architecture
        - Rotary positional embeddings
        - Sliding attention layers
        - Grouped query
        - Output Layer
    5. Pre-training
    6. Infrence

"""
# Imports
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm


def step1_loading_dataset(hf_path: str = "roneneldan/TinyStories"):
    try:
        from datasets import load_dataset
        return load_dataset(hf_path)
    except Exception as e:
        # logging.error(f"Unable to Intialize the {hf_path} dataset.")
        raise RuntimeError(
            f"Could not initialize Hugging Face dataset: {hf_path}"
        ) from e

class DatasetTokenization:
    def __init__(self, ds: Dataset = step1_loading_dataset()):
        self.encode = tiktoken.get_encoding("gpt2")

    def _process(self, example):
        input_ids = self.encode.encode_ordinary(example['text']) # encode_ordinary igonres any special tokens
        return {
            "input_ids": input_ids,
            "len": len(ids)
        }    

    def run(self):
        if not os.path.exists("train.bin"): # if train.bin doesn't exists it will create one
            tokenized = ds.map(
                self._process,
                remove_columns = ['text'],
                desc = "Tokenizing the splits",
                num_proc = 2, ## number of cpu to use
            )

            ## concatenate all the input_ids in each dataset into one large file we can use for training
            for split, dset in tokenized.items():
                arr_len = np.sum(dset['len'], dtype = np.uint64)
                filename = f"{split}.bin"
                dtype = np.uint16 ## can do this since enc.max_token_value(vocab size) == 50256 is < 2**16
                arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
                total_batches = 10

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc = f"writing {filename}"):
                    # batch together samples for faster write 
                    batch = dset.shard(
                        num_shards = total_batches,
                        index = batch_idx,
                        contiguous = True
                    ).with_format("numpy")
                    arr_batch = np.concatenate(batch['ids'])
                    # write into mmap
                    arr[idx: idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()

if __name__ == "__main__":
    step2_obj = DatasetTokenization()
    step2_obj.run()          