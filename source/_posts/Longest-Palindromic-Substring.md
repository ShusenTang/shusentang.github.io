---
title: [LeetCode]Longest Palindromic Substring(最长回文子串)
date: 2018-12-01 22:23:15
toc: true
categories: 
- LeetCode
tags:
- 动态规划
- 马拉车算法
- 字符串
---

# 1. 问题描述
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.
给定一个字符串s，找出s中的最长回文子串。s的长度不超过1000.

Example 1:
>Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:
> Input: "cbbd"
Output: "bb"

# 2. 题解
## 2.1 暴力法
### 2.1.1 思路
暴力穷举所有子字符串的可能，然后依次按位判断其是否是回文。虽然其时间复杂度很高，但它对空间的要求很低。
### 2.1.2 复杂度
空间复杂度显然为 $O(1)$ ；
穷举所有字符串复杂度为 $O(n^2)$ ，再加上判断是否是回文的，所以总的时间复杂度为 $O(n^3)$ 。

### 2.1.3 代码
代码就不给了，给了你也不会看😀

## 2.2 动态规划
### 2.2.1 思路
我们知道，当某个字符串是回文的时候，在它两边同时增加一个相同字符后肯定依然是回文的，例如 bccb -> abccba。基于这个特点，我们可以将暴力法中的判断是否为回文这一步用动态规划解决。

具体的，我们可以先把所有长度为1的子字符串计算出来，这些必定是回文的；然后计算出所有长度为2的子字符串并判断是否为回文。到长度为3的时候，我们就可以利用上次的计算结果：如果中心对称的短字符串不是回文，那长字符串也不是，如果短字符串是回文，那就要看长字符串两头是否一样。这样，一直到长度最大的子字符串，我们就把整个字符串集穷举完了。在这个过程中用一个 n x n 的二维bool型数组`isOK`记录是否会回文，其中n是字符串s的长度，`isOK[i][j] == true`就表示子串`s[i...j]`是回文的。

### 2.2.2 复杂度
由于要申请一个二维数组`isOK`, 所以空间复杂度为 $O(n^2)$ ;
由于使用动态规划，时间复杂度从暴力法的 $O(n^3)$ 减少到 $O(n^2)$。
实测:
> Runtime: 564 ms, faster than 9.98% of C++ online submissions for Longest Palindromic Substring.

### 2.2.3 代码
``` C++
class Solution {
public:
    string longestPalindrome(string s) {
        if(!s.size()) return "";
        vector<vector<bool>>isOK(s.size(), vector<bool>(s.size(), false));
        string res = "";
        for(int len = 1; len <= s.size(); len++){ // len代表子串长度
            for(int start = 0; start + len <= s.size(); start++){ // start代表子串起始位置
                if(len == 1) isOK[start][start] = true;
                else if(len == 2) isOK[start][start + 1] = (s[start] == s[start + 1]);
                else isOK[start][start + len - 1] = isOK[start + 1][start + len - 2] \
                    && s[start] == s[start + len - 1];
                
                res = isOK[start][start + len - 1] && len > res.size() ? s.substr(start, len): res;
            } 
        } 
        return res;
    }
};
```

## 2.3 中心扩散法
### 2.3.1 思路
上面讲的动态规划虽然优化了时间，但也浪费了空间。实际上我们并不需要一直存储所有子字符串的回文情况，我们需要知道的只是中心对称的较小一层是否是回文。所以如果我们从小到大连续以某点为个中心的所有子字符串进行计算，就能省略这个空间。 
需要注意的是，由于中心对称有两种情况:
1. 长度为奇数，则以中心字母对称；
2. 长度为偶数，则以两个字母中间为对称。

所以我们要分别计算这两种对称情况。

### 2.3.2 复杂度
相较于2.2的动态规划，空间复杂度缩小为 $O(1)$；
时间复杂度一样，都是 $O(n^2)$。

### 2.3.3 代码


## 2.4 马拉车(Manacher)算法