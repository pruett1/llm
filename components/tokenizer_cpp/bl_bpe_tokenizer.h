#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <unordered_map>
#include <utility>

// hash function for vector<int> to be used in unordered_map
// went with xor folding bc vectors are in general short (most length 1-2, special tokens will be longer but rare and still likely short)
struct vector_hash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for(auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& v) const {
        return std::hash<int>()(v.first) ^ (std::hash<int>()(v.second) << 1);
    }
};

// trie node struct to speed up special token matching
struct TrieNode {
    std::unordered_map<int, TrieNode*> children;
    int tokenId = -1;

    ~TrieNode() {
        for (auto &[_, child] : children) {
            delete child;
        }
    }
};

class BlBPETokenizer {
    public:
        BlBPETokenizer(int vocabSize, const std::vector<std::string>& specialTokens);

        ~BlBPETokenizer();

        void train(const std::vector<std::string>& texts);

        std::vector<int> encode(std::string text);

        std::string decode(std::vector<int> tokens);

    private:
        int vocabSize;
        std::unordered_map<std::vector<int>, int, vector_hash> vocab;
        std::unordered_map<int, std::vector<int>> invVocab;

        TrieNode* specialTokenRoot;
        
        std::vector<std::vector<int>> merges;

        std::pair<int, int> getStats(const std::vector<std::vector<int>>& tokens) const;

        void mergeVocab(const std::pair<int, int>& pair, std::vector<std::vector<int>>& tokens);

        std::vector<std::vector<int>> corpusToTokens(const std::vector<std::string>& texts) const;

        void getBytes(int token, std::vector<uint8_t>& outBytes);
};