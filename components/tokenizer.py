from collections import Counter

class CharTokenizer():
    # initial single character tokens
    def __init__(self):
        self.vocab = Counter()
        self.inv_vocab = {}
    
    def encode(self, text: str) -> list[int]:
        self.vocab = Counter(text)
        #handle duplicates from counter by assigning new ids
        cnt = 0
        for k in self.vocab.keys():
            self.vocab[k] = cnt
            cnt += 1
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return [self.vocab[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.inv_vocab[t] for t in tokens])
    
#byte-level BPE tokenizer because training on mix of text and code
#allowing special token to be added for output labeling
class BlBPETokenizer():
    def __init__(self, vocab_size: int = 10000, special_tokens: list[str] = []):
        self.special = special_tokens
        self.vocab_size = vocab_size
        # the vocab maps are ints because the combined byte pairs will otherwise throw issues bc > 255
        self.vocab = {i: i for i in range(256)} # initial byte-level vocab
        self.inv_vocab = {i: i for i in range(256)}

        #add special tokens to vocab
        for token in special_tokens:
            idx = len(self.vocab)
            token_bytes = list(token.encode('utf-8'))
            self.vocab[tuple(token_bytes)] = idx
            self.inv_vocab[idx] = tuple(token_bytes)

        self.merges = []
        pass

    def get_stats(self, tokens: list[list[bytes]]) -> Counter:
        #count freq of each adjacent byte pair in the tokenized data
        pairs = Counter()
        for seq in tokens:
            for i in range(len(seq) - 1):
                pair = seq[i], seq[i+1]
                pairs[pair] += 1

        return pairs
    
    def merge_vocab(self, pair: tuple[int, int], tokens: list[list[bytes]]) -> list[list[bytes]]:
        #apply merge to token sequences
        new_tokens = []
        new_id = self.vocab[pair]

        for seq in tokens:
            merged_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                    merged_seq.append(new_id)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_tokens.append(merged_seq)
        return new_tokens
    
    def corpus_to_tokens(self, texts: list[str]) -> list[list[int]]:
        #convert texts to list of byte tokens
        special_map = {tuple(s.encode('utf-8')): self.vocab[tuple(s.encode('utf-8'))] for s in self.special}
        tokens = []
        for text in texts:
            byte_seq = list(text.encode('utf-8'))
            i = 0
            while i < len(byte_seq):
                replaced = False
                for s_bytes, s_id in special_map.items():
                    if byte_seq[i:i+len(s_bytes)] == list(s_bytes):
                        byte_seq = byte_seq[:i] + [s_id] + byte_seq[i+len(s_bytes):]
                        replaced = True
                        break
                if not replaced:
                    i += 1
            tokens.append(byte_seq)
        return tokens

    def train(self, texts: list[str]):
        #train BPE tokenizer on list of provided texts
        tokens = self.corpus_to_tokens(texts)
        
        while len(self.vocab) < self.vocab_size:
            stats = self.get_stats(tokens)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_token = best_pair

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = new_token
                self.merges.append(best_pair)

            tokens = self.merge_vocab(best_pair, tokens)

    
    def encode(self, text: str) -> list[int]:
        #convert text to byte tokens and apply merges
        tokens = self.corpus_to_tokens([text])[0]

        for pair in self.merges:
            i = 0    
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(self.vocab[pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def get_bytes(self, token) -> bytes:
        # recursively expand token into bytes
        val = self.inv_vocab[token]
        if isinstance(val, int):
            return bytes([val])

        out = b""
        for sub_id in val:
            out += self.get_bytes(sub_id)
        return out

    def decode(self, tokens: list[int]) -> str:
        #convert tokens back to text
        byte_seq = b"".join([self.get_bytes(t) for t in tokens])
        return byte_seq.decode('utf-8', errors='replace')

# tokenizer = BlBPETokenizer(vocab_size=290, special_tokens=["###OUTPUT###", "###END###"])
# texts = ["test 123 ###OUTPUT### some out ###END###"]
# tokenizer.train(texts)
# tokens = tokenizer.encode("test hi ###OUTPUT### some output here 12:34 ###END### next thing now ###OUTPUT### another output ###END###")
# print(tokens)
# print(tokenizer.decode(tokens))
