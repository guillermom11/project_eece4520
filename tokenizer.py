from collections import defaultdict

class BPE:
    def __init__(self):
        self.vocab = defaultdict(int)
        self.merges = {}

    def build_vocab(self, data):
        # The initial vocabulary consists of the given words, where each word is represented
        # as a sequence of symbols separated by spaces, e.g., ['L o w', 'n e w']
        for word in data:
            self.vocab[' '.join(word)] += 1

    def count_pairs(self):
        # Count the frequency of symbol pairs in the vocabulary
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, best_pair):
        # Merge symbols in the vocabulary based on the best pair
        new_vocab = {}
        bigram = ' '.join(best_pair)
        for word in self.vocab:
            new_word = word.replace(bigram, ''.join(best_pair))
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab

    def train_bpe(self, data, num_merges):
        # Train the BPE model using the input data
        self.build_vocab(data)
        for i in range(num_merges):
            pairs = self.count_pairs()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = i
            self.merge_vocab(best_pair)

    def tokenize(self, word):
        # Tokenize the input word using the learned merge operations
        tokens = list(word)

        while True:
            # Find a merge in the word with the lowest index
            lowest_merge_index = float('inf')
            lowest_merge = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    if self.merges[pair] < lowest_merge_index:
                        lowest_merge_index = self.merges[pair]
                        lowest_merge = pair

            if not lowest_merge:
                break

            # Apply the lowest merge to the word
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

        # Convert tokens to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id["<UNK>"])

        return ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        text = ''.join(tokens)
        return text