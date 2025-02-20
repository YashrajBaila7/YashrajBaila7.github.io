import os
import json
import torch
from torch.utils.data import Dataset


class BPEDataset(Dataset):
    """
    Dataset class for text data with Byte Pair Encoding (BPE) support.
    """

    def __init__(self, data, block_size, output_dir=None):
        """
        Initialize the BPEDataset.

        Args:
            data (str): Text data.
            block_size (int): Size of the text blocks.
            output_dir (str, optional): Output directory to save encoder.json and merges.json.
        """
        # Ensure output directory exists if provided
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get unique characters from the data
        chars = sorted(list(set(data)))
        data_size, unique_char_count = len(data), len(chars)
        print('Data has %d characters, %d unique.' %
              (data_size, unique_char_count))

        # Use the unique character count as the initial vocabulary size.
        # If you prefer to use a fixed size (e.g., 256) you can modify this accordingly.
        self.vocab_size = unique_char_count
        self.block_size = block_size
        self.data = data

        # Create character to index and index to character mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Build the initial BPE vocabulary from the unique characters.
        self.bpe_vocab = {i: ch.encode("utf-8") for i, ch in enumerate(chars)}
        # Will store merge rules as {(p0, p1): new_token_id}
        self.bpe_merges = {}

        if output_dir:
            encoder_file = os.path.join(output_dir, 'encoder.json')
            try:
                with open(encoder_file, 'w', encoding='utf-8') as f:
                    json.dump(self.stoi, f)
            except Exception as e:
                print(f"Error writing encoder file: {e}")

    def get_vocab_size(self):
        """
        Get the vocabulary size.

        Returns:
            int: Vocabulary size.
        """
        return self.vocab_size

    def get_block_size(self):
        """
        Get the block size.

        Returns:
            int: Block size.
        """
        return self.block_size

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing input and target tensors.
        """
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def train_bpe(self, target_vocab_size):
        """
        Train BPE on the dataset.

        Args:
            target_vocab_size (int): Target vocabulary size after BPE merges.
        """
        # Convert the text to token IDs using self.stoi
        tokens = [self.stoi[ch] for ch in self.data]
        num_merges = target_vocab_size - self.vocab_size

        for i in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = self.vocab_size + i
            print(f"Merging {pair} into a new token {idx}")
            tokens = self._merge(tokens, pair, idx)
            self.bpe_merges[pair] = idx

        # Update bpe_vocab sequentially so that dependencies are met
        for new_id in range(self.vocab_size, target_vocab_size):
            for pair, token_id in self.bpe_merges.items():
                if token_id == new_id:
                    self.bpe_vocab[new_id] = self.bpe_vocab[pair[0]
                                                            ] + self.bpe_vocab[pair[1]]
                    break

        # Update the vocabulary size to reflect the new tokens
        self.vocab_size = target_vocab_size

        if self.output_dir:
            # Save the BPE merges to a JSON file with proper serialization of tuple keys
            merges_file = os.path.join(self.output_dir, 'merges.json')
            serializable_merges = {
                f"{pair[0]},{pair[1]}": idx for pair, idx in self.bpe_merges.items()}
            try:
                with open(merges_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_merges, f)
            except Exception as e:
                print(f"Error writing merges file: {e}")

        print("BPE training complete.")

    def _get_stats(self, tokens):
        """
        Calculate the frequency of adjacent token pairs.

        Args:
            tokens (list): List of token IDs.

        Returns:
            dict: Frequency counts of adjacent token pairs.
        """
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, tokens, pair, idx):
        """
        Merge a token pair into a new token.

        Args:
            tokens (list): List of token IDs.
            pair (tuple): Token pair to merge.
            idx (int): New token ID.

        Returns:
            list: Updated token list.
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _encode(self, s):
        """
        Encode a string into a list of integers using the BPE rules.

        Args:
            s (str): Input string.

        Returns:
            list: List of encoded token IDs.
        """
        # Instead of using raw bytes, tokenize based on the character mapping
        tokens = [self.stoi[ch] for ch in s]
        while len(tokens) > 1:
            stats = self._get_stats(tokens)
            # Use the merge order as defined in bpe_merges
            pair = min(
                stats, key=lambda p: self.bpe_merges.get(p, float("inf")))
            if pair not in self.bpe_merges:
                break
            idx = self.bpe_merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def _decode(self, tokens):
        """
        Decode a list of token IDs into a string.

        Args:
            tokens (list): List of token IDs.

        Returns:
            str: Decoded string.
        """
        decoded_bytes = b"".join(self.bpe_vocab[t] for t in tokens)
        return decoded_bytes.decode("utf-8", errors="replace")


class Encoder:
    """
    Encoder class that loads an encoder mapping and BPE merge rules from JSON files,
    then provides methods for encoding text into token IDs and decoding token IDs back to text.
    """

    def __init__(self, encoder_path, merges_path):
        """
        Initialize the Encoder by loading the encoder and merges JSON files.

        Args:
            encoder_path (str): Path to the encoder JSON file (character-to-token mapping).
            merges_path (str): Path to the merges JSON file (serialized BPE merge rules).
        """
        # Load encoder mapping (character -> token id)
        try:
            with open(encoder_path, 'r', encoding='utf-8') as f:
                self.stoi = json.load(f)
            # Ensure that token IDs are integers
            self.stoi = {ch: int(idx) for ch, idx in self.stoi.items()}
        except Exception as e:
            raise IOError(f"Error reading encoder file: {e}")

        # Create reverse mapping (token id -> character)
        self.itos = {v: ch for ch, v in self.stoi.items()}

        # Load merges (serialized as string keys "p0,p1") and convert back to tuple keys.
        self.merges = {}
        try:
            with open(merges_path, 'r', encoding='utf-8') as f:
                merges_serializable = json.load(f)
            for key, value in merges_serializable.items():
                p0, p1 = key.split(',')
                self.merges[(int(p0), int(p1))] = int(value)
        except Exception as e:
            print(f"Error reading merges file: {e}")

        # Build the initial BPE vocabulary.
        # The initial vocabulary is simply the mapping of token IDs (for characters) to their encoded bytes.
        self.bpe_vocab = {i: ch.encode("utf-8") for i, ch in self.itos.items()}

        # Update the BPE vocabulary by processing merge rules in order of increasing token ID.
        # This ensures that when a merge rule is applied, its constituent tokens have already been defined.
        for (p0, p1), token_id in sorted(self.merges.items(), key=lambda item: item[1]):
            self.bpe_vocab[token_id] = self.bpe_vocab[p0] + self.bpe_vocab[p1]

    def encode(self, text):
        """
        Encode a string into a list of token IDs using the loaded BPE merge rules.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list: A list of token IDs representing the BPE-encoded text.
        """
        # Start with a simple character-level tokenization based on self.stoi.
        tokens = [self.stoi[ch] for ch in text]

        # Iteratively apply merge rules until no applicable merge is found.
        while True:
            stats = self._get_stats(tokens)
            if not stats:
                break
            # Select the pair with the lowest merge order (if not present, returns infinity).
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            new_token = self.merges[pair]
            tokens = self._merge(tokens, pair, new_token)
        return tokens

    def _get_stats(self, tokens):
        """
        Compute the frequency of adjacent token pairs in a token list.

        Args:
            tokens (list): List of token IDs.

        Returns:
            dict: Dictionary with keys as token pairs (tuple) and values as their frequencies.
        """
        stats = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge(self, tokens, pair, new_token):
        """
        Merge occurrences of a specific token pair into a new token.

        Args:
            tokens (list): Current list of token IDs.
            pair (tuple): The token pair to merge.
            new_token (int): The new token ID that replaces the pair.

        Returns:
            list: Updated list of token IDs after merging.
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def decode(self, tokens):
        """
        Decode a list of token IDs back into a string.

        Args:
            tokens (list): List of token IDs (as produced by the encode method).

        Returns:
            str: The decoded text.
        """
        # Reconstruct the original byte sequence using the BPE vocabulary.
        decoded_bytes = b"".join(self.bpe_vocab.get(token, b"")
                                 for token in tokens)
        return decoded_bytes.decode("utf-8", errors="replace")


class FinetuneDataset(Dataset):
    """
    PyTorch dataset for fine-tuning language models using BPE tokenization.

    This dataset pre-encodes the entire text using the provided Encoder and then creates
    fixed-length token sequences (input-target pairs) for training. This ensures that every
    sample has the same length, avoiding issues with variable-length outputs from on-the-fly encoding.
    """

    def __init__(self, data, encoder_path, merges_path, block_size):
        """
        Initialize the FinetuneDataset.

        Args:
            data (str): The raw input text to be tokenized.
            encoder_path (str): Path to the encoder JSON file (character-to-token mapping).
            merges_path (str): Path to the merges JSON file (BPE merge rules).
            block_size (int): The desired number of tokens in each input sample.
                               Each sample will have block_size tokens as input and block_size tokens as target.
        """
        # Instantiate the pre-trained Encoder using the provided file paths.
        self.encoder = Encoder(encoder_path, merges_path)
        self.block_size = block_size

        # Pre-encode the entire text once to ensure consistent and fixed-length token sequences.
        self.encoded_data = self.encoder.encode(data)
        if len(self.encoded_data) < block_size + 1:
            raise ValueError(
                "The encoded data is shorter than the block size + 1.")

    def __len__(self):
        """
        Returns the number of samples that can be drawn from the tokenized data.
        """
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): The starting index for the token sequence.

        Returns:
            tuple: A tuple (x, y) where:
                   - x is a tensor of token IDs of length block_size (input sequence),
                   - y is a tensor of token IDs of length block_size (target sequence, which is the input shifted by one token).
        """
        # Slice the pre-encoded data to create fixed-length input-target pairs.
        x = self.encoded_data[idx: idx + self.block_size]
        y = self.encoded_data[idx + 1: idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def count_unique_characters(file_path):
    """
    Count the number of unique characters in a text file using UTF-8 encoding.

    Args:
        file_path (str): Path to the text file.

    Returns:
        int: Number of unique characters in the file.
    """
    try:
        # Read the file content as a string
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Use a set to find unique characters
        unique_chars = set(text)
        return len(unique_chars)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return 0
