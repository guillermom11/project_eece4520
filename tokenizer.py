from collections import defaultdict

class BPE:
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BPE, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'vocab'):  # Ensure __init__ is only executed once
            self.vocab = defaultdict(int)
            self.merges = {}
            self.token_to_id = {}  # Define token-to-ID mapping
            self.id_to_token = {}  # Define ID-to-token mapping

    def build_vocab(self, data):
        for word in data:
            self.vocab[' '.join(word)] += 1

    def count_pairs(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, best_pair):
        new_vocab = {}
        bigram = ' '.join(best_pair)
        for word in self.vocab:
            new_word = word.replace(bigram, ''.join(best_pair))
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab

    def train_bpe(self, data, num_merges):
        self.build_vocab(data)
        for i in range(num_merges):
            pairs = self.count_pairs()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = i
            self.merge_vocab(best_pair)

    def tokenize(self, word):
        tokens = list(word)
        while True:
            lowest_merge_index = float('inf')
            lowest_merge = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges and self.merges[pair] < lowest_merge_index:
                    lowest_merge_index = self.merges[pair]
                    lowest_merge = pair

            if not lowest_merge:
                break

            for i in range(len(tokens) - 1):
                if (tokens[i], tokens[i + 1]) == lowest_merge:
                    tokens[i:i + 2] = [''.join(tokens[i:i + 2])]
                    break

        return tokens

    def encode(self, text):
        words = text.lower().split()
        tokens = []
        for word in words:
            word_tokens = self.tokenize(word)
            tokens.extend(word_tokens)

        ids = [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0)) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        return ''.join(tokens)
