#include "bl_bpe_tokenizer.h"

BlBPETokenizer::BlBPETokenizer(int vocabSize, const std::vector<std::string>& specialTokens) {
    this->vocabSize = vocabSize;
    this->vocab = {};
    vocab.reserve(vocabSize);
    this->invVocab = {};
    invVocab.reserve(vocabSize);
    this->specialTokenMap = {};
    specialTokenMap.reserve(specialTokens.size());
    this->merges = {};
    merges.reserve(vocabSize-256-specialTokens.size()); //vocab_size - 256 bytes - #special tokens added
    this->specialTokenRoot = new TrieNode();

    for (int i = 0; i < 256; i++) {
        std::vector<int> byte = {i};
        vocab.emplace(byte, i);
        invVocab.emplace(i, byte);
    }

    for (size_t i = 0; i < specialTokens.size(); i++) {
        int idx = static_cast<int>(vocab.size());
        std::vector<int> tokenBytes;
        for (unsigned char c : specialTokens[i]) {
            tokenBytes.push_back(static_cast<int>(c));
        }
        vocab.emplace(tokenBytes, idx);
        invVocab.emplace(idx, tokenBytes);
        specialTokenMap.emplace(specialTokens[i], idx);
        TrieNode* node = specialTokenRoot;
        for (int b : tokenBytes) {
            if (node->children.find(b) == node->children.end()) {
                node->children[b] = new TrieNode();
            }
            node = node->children[b];
        }
        node->tokenId = idx;
    }
}

BlBPETokenizer::~BlBPETokenizer() {
    delete specialTokenRoot;
}

int BlBPETokenizer::getSpecialTokenId(std::string token) {
    return specialTokenMap.at(token);
}

std::pair<int, int> BlBPETokenizer::getStats(const std::vector<std::vector<int>>& tokens) const {
    std::unordered_map<std::pair<int, int>, int, pair_hash> pairs;
    std::pair<int, int> bestPair = {-1, -1};
    int maxFreq = 0;

    //count freq of each adjacent byte pair in the tokenized data
    for (const std::vector<int>& seq: tokens) {
        for (size_t i = 0; i < seq.size() - 1; i++) {
            std::pair<int, int> pairKey(seq[i], seq[i+1]);
            int freq = ++pairs[pairKey];
            if (freq > maxFreq) {
                bestPair = pairKey;
                maxFreq = freq;
            }
        }
    }
    return bestPair;
}

void BlBPETokenizer::mergeVocab(const std::pair<int, int>& pair, std::vector<std::vector<int>>& tokens) {
    int newId = vocab.at(std::vector<int>{pair.first, pair.second});

    for (auto& seq: tokens) {
        size_t write_pos = 0;

        for (size_t read_pos = 0; read_pos < seq.size(); ) {
            if (read_pos < seq.size() - 1 && seq[read_pos] == pair.first && seq[read_pos + 1] == pair.second) {
                seq[write_pos++] = newId;
                read_pos += 2;
            }
            else {
                seq[write_pos++] = seq[read_pos++];
            }
        }
        seq.resize(write_pos);
    }
}

std::vector<std::vector<int>> BlBPETokenizer::corpusToTokens(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> tokens;
    tokens.reserve(texts.size());

    for (const std::string& text: texts) {
        std::vector<int> byteSeq;
        byteSeq.reserve(text.size());
        
        int i = 0;
        while (i < (int)text.size()) {
            TrieNode* node = specialTokenRoot;
            int j = i;
            int matchedId = -1;
            int matchedLen = 0;

            while (j < (int)text.size()) {
                unsigned char c = text[j];
                auto it = node->children.find(static_cast<int>(c));
                if (it == node->children.end()) {
                    break;
                }
                node = it->second;
                j++;

                if (node->tokenId != -1) {
                    matchedId = node->tokenId;
                    matchedLen = j - i;
                }
            }

            if (matchedId != -1) {
                byteSeq.push_back(matchedId);
                i += matchedLen;
            } else {
                byteSeq.push_back(static_cast<int>(text[i]));
                i++;
            }
        }

        tokens.push_back(std::move(byteSeq));
    }

    return tokens;
}

void BlBPETokenizer::train(const std::vector<std::string>& texts) {
    std::vector<std::vector<int>> tokens = corpusToTokens(texts);

    while ((int)vocab.size() < vocabSize) {
        std::pair<int, int> bestPair = getStats(tokens);
        if (bestPair.first == -1) break;

        std::vector<int> newToken = {bestPair.first, bestPair.second};
        
        if (vocab.find(newToken) == vocab.end()) {
            int newId = vocab.size();
            vocab.emplace(newToken, newId);
            invVocab.emplace(newId, newToken);
            merges.push_back(newToken);
        }
        mergeVocab(bestPair, tokens);
    }
}

std::vector<int> BlBPETokenizer::encode(std::string text) {
    std::vector<int> tokens = corpusToTokens(std::vector<std::string>{text})[0];
    std::vector<int> newTokens(tokens.size());

    std::vector<int>* in = &tokens;
    std::vector<int>* out = &newTokens;

    for (auto& pair: merges) {
        int pairInd = vocab.at(pair);
        size_t write_pos = 0;

        for (size_t read_pos = 0; read_pos < in->size(); ) {
            if (read_pos < in->size() - 1 && (*in)[read_pos] == pair[0] && (*in)[read_pos + 1] == pair[1]) {
                (*out)[write_pos++] = pairInd;
                read_pos += 2;
            } else {
                (*out)[write_pos++] = (*in)[read_pos++];
            }
        }
        out->resize(write_pos);
        std::swap(in, out);
    }
    
    return std::move(*in);
}

void BlBPETokenizer::getBytes(int token, std::vector<uint8_t>& outBytes) {
    auto it = invVocab.find(token);
    if (it == invVocab.end()) return;

    const auto& vals = it->second;
    if (vals.size() == 1) {
        outBytes.push_back(static_cast<uint8_t>(vals[0]));
    }
    else {
        for (int b: vals) {
            getBytes(b, outBytes);
        }
    }
}

std::string BlBPETokenizer::decode(std::vector<int> tokens) {
    std::vector<uint8_t> bytes;
    bytes.reserve(tokens.size() * 4);

    for (int token : tokens) {
        getBytes(token, bytes);
    }
    return std::string(bytes.begin(), bytes.end());
}