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

class BlBPETokenizer {
    public:
        BlBPETokenizer(int vocabSize, const std::vector<std::string>& specialTokens);

        void train(const std::vector<std::string>& texts);

        std::vector<int> encode(std::string text);

        std::string decode(std::vector<int> tokens);

    private:
        int vocabSize;
        std::unordered_map<std::vector<int>, int, vector_hash> vocab;
        std::unordered_map<int, std::vector<int>> invVocab;

        std::unordered_map<std::vector<int>, int, vector_hash> specialTokensMap;
        
        std::vector<std::vector<int>> merges;

        std::map<std::pair<int, int>, int> getStats(const std::vector<std::vector<int>>& tokens) const;

        std::vector<std::vector<int>> mergeVocab(const std::pair<int, int>& pair, const std::vector<std::vector<int>>& tokens);

        std::vector<std::vector<int>> corpusToTokens(const std::vector<std::string>& texts) const;

        std::vector<uint8_t> getBytes(int token);
};