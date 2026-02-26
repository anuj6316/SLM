import re, collections
from pprint import pprint
import logging

logger = logging.getLogger(__name__)

def get_stats(vocab):
    """
    count occurrences of all adjacent pairs of symbols.
    """
    pairs = collections.defaultdict(int)
    logger.info(f"=== Pairs ===\n{pairs}")
    print(f"=== Pairs ===\n{pairs}")

    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    Merge all occurrences of the most frequent pair in the vocabulary.
    """
    v_out = {}
    bigram = re.escape(" ".join(pair))
    # Match the pair only if it's surrounded by spaces or string boundaries
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

    for word in v_in:
        # Replace "t h" with "th"
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

if __name__ == "__main__":
    # 1. Initial Vocabulary (Words split into characters + end-of-word marker)
    vocab = {
        'l o w </w>': 5,
        'l o w e r </w>': 2,
        'n e w e s t </w>': 6,
        'w i d e s t </w>': 3
    }

    # 2. Iterative Merging
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        # find the most frequent pair
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

        print(f"Iteration {i + 1}: Best pair '{best}' merged {pairs[best]} times.")
        print(f"Current Vocab: {vocab}\n")