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
    
# tokenizer = CharTokenizer()
# print(type(tokenizer))

# text = '{"desc": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.\nYou may assume that each input would have exactly one solution, and you may not use the same element twice.\nYou can return the answer in any order.", "examples": ["Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nExplanation: Because nums[0] + nums[1] == 9, we return [0, 1].", "Input: nums = [3,2,4], target = 6\nOutput: [1,2]", "Input: nums = [3,3], target = 6\nOutput: [0,1]"], "constraints": ["2 <= nums.length <= 104", "-109 <= nums[i] <= 109", "-109 <= target <= 109", "Only one valid answer exists.", "", "Follow-up:", "O(n2)"], "solution": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        pair_idx = {}\n\n        for i, num in enumerate(nums):\n            if target - num in pair_idx:\n                return [i, pair_idx[target - num]]\n            pair_idx[num] = i\n"'
# tokens = tokenizer.encode(text)
# print(len(tokens))
# print(tokenizer.vocab)
# print(len(tokenizer.vocab))
# print(text == tokenizer.decode(tokens))