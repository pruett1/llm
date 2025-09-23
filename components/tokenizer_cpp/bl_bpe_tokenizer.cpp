#include "bl_bpe_tokenizer.h"

BlBPETokenizer::BlBPETokenizer(int vocabSize, const std::vector<std::string>& specialTokens) {
    this->vocabSize = vocabSize;
    this->vocab = {};
    this->invVocab = {};
    this->merges = {};
    this->specialTokensMap = {};

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
        specialTokensMap.emplace(tokenBytes, idx);
    }
}

std::map<std::pair<int, int>, int> BlBPETokenizer::getStats(const std::vector<std::vector<int>>& tokens) const {
    std::map<std::pair<int, int>, int> pairs;
    //count freq of each adjacent byte pair in the tokenized data
    for (const std::vector<int>& seq: tokens) {
        for (size_t i = 0; i < seq.size() - 1; i++) {
            int first = seq[i];
            int second = seq[i + 1];
            std::pair<int, int> pairKey(first, second);
            pairs[pairKey]++;
        }
    }
    return pairs;
}

std::vector<std::vector<int>> BlBPETokenizer::mergeVocab(const std::pair<int, int>& pair, const std::vector<std::vector<int>>& tokens) {
    std::vector<std::vector<int>> newTokens;
    int newId = vocab[std::vector<int>{pair.first, pair.second}];

    for (const std::vector<int>& seq: tokens) {
        std::vector<int> newSeq;
        int i = 0;
        while (i < (int)seq.size()) {
            if (i < (int)seq.size() - 1 && seq[i] == pair.first && seq[i+1] == pair.second) {
                newSeq.push_back(newId);
                i += 2;
            }
            else{
                newSeq.push_back(seq[i]);
                i++;
            }
        }
        newTokens.push_back(newSeq);
    }
    return newTokens;
}

std::vector<std::vector<int>> BlBPETokenizer::corpusToTokens(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> tokens;

    for (const std::string& text: texts) {
        std::vector<int> byteSeq;
        for (unsigned char c: text) {
            byteSeq.push_back(static_cast<int>(c));
        }
        int i = 0;
        while (i < (int)byteSeq.size()) {
            bool replaced = false;
            for (const auto& [specBytes, specId] : specialTokensMap) {
                int tokenSize = specBytes.size();
                std::vector<int> subVec(byteSeq.begin() + i, byteSeq.begin() + i + tokenSize);
                if (subVec == specBytes) {
                    byteSeq.erase(byteSeq.begin() + i, byteSeq.begin() + i + tokenSize);
                    byteSeq.insert(byteSeq.begin() + i, specId);
                    replaced = true;
                    break;
                }
            }
            if (!replaced) {
                i++;
            }
        }
        tokens.push_back(byteSeq);
    }

    return tokens;
}

void BlBPETokenizer::train(const std::vector<std::string>& texts) {
    std::vector<std::vector<int>> tokens = corpusToTokens(texts);

    while ((int)vocab.size() < vocabSize) {
        std::map<std::pair<int, int>, int> stats = getStats(tokens);
        if (stats.empty()) {
            break;
        }
        std::pair<int, int> bestPair;
        int maxFreq = 0;
        for (const auto& [pair, freq]: stats) {
            if (freq > maxFreq) {
                bestPair = pair;
                maxFreq = freq;
            }
        }
        std::vector<int> newToken(bestPair.first, bestPair.second);
        
        if (vocab.find(newToken) == vocab.end()) {
            vocab.emplace(newToken, vocab.size());
            invVocab.emplace(vocab.size() - 1, newToken);
            merges.push_back(newToken);
        }
        tokens = mergeVocab(bestPair, tokens);
    }
}

std::vector<int> BlBPETokenizer::encode(std::string text) {
    std::vector<int> tokens = corpusToTokens(std::vector<std::string>{text})[0];

    for (auto& pair: merges) {
        int i = 0;
        std::vector<int> new_tokens;
        while (i < (int)tokens.size()) {
            if (i < (int)tokens.size() - 1 && tokens[i] == pair[0] && tokens[i+1] == pair[1]) {
                new_tokens.push_back(vocab[pair]);
                i += 2;
            }
            else {
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        tokens = new_tokens;
    }
    return tokens;
}

std::vector<uint8_t> BlBPETokenizer::getBytes(int token) {
    std::vector<uint8_t> bytes;
    if (invVocab.find(token) != invVocab.end() && invVocab[token].size() == 1) {
        bytes.push_back(static_cast<uint8_t>(invVocab[token][0]));
        return bytes;
    }
    else if (invVocab.find(token) != invVocab.end()) {
        for (int b : invVocab[token]) {
            std::vector<uint8_t> subBytes = getBytes(b);
            bytes.insert(bytes.end(), subBytes.begin(), subBytes.end());
        }
    }

    return bytes;
}

std::string BlBPETokenizer::decode(std::vector<int> tokens) {
    std::vector<uint8_t> bytes;
    for (int token : tokens) {
        std::vector<uint8_t> tokenBytes = getBytes(token);
        bytes.insert(bytes.end(), tokenBytes.begin(), tokenBytes.end());
    }
    return std::string(bytes.begin(), bytes.end());
}