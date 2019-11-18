---
title: 动态规划之股票买卖系列
date: 2019-11-03 17:53:05
toc: true
mathjax: true
categories: 
- LeetCode
tags:
- 动态规划
---

<center>
<img src="./Buy-and-Sell-Stock/cover.png" width="500" class="full-image">
</center>

股票买卖系列是动态规划的经典题目，Leetcode上有六道关于股票买卖相关的问题，本文对这六道题作一个分析与总结。

<!-- more -->

# 1. 总体分析
## 1.1 题意
给定一个大小为n的数组`prices`代表连续n天某支股票的股价，`prices[i]`即第i天的股价，且必须在买进后才能卖出。再给定一些限制条件，问最大收益。一般有如下几种限制条件，对于LeetCode上六道题：
1. 买卖一次: [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
2. 不限买卖次数: [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
3. 买卖两次: [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)
4. 买卖k次:[188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)
5. 带有冷却的股票买卖问题:[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
6. 带有交易费用的股票买卖问题:[714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

## 1.2 分析
前两种情况比较简单，属于LeetCode中的easy题，所以我们考虑普遍情况：可买卖k次。动态规划的关键在于状态以及状态转移方程如何定义。首先考虑影响状态的变量：
* 当前处于第几天；
* 已经交易的次数；
* 手头是否持有股票；

即根据手头是否持有股票，我们定义两个二维数组来定义状态：
```
dp0[i][j]: 第i天结束，已有j次买卖，手头没有股票时的最大利润
dp1[i][j]: 第i天结束，已有j次买卖，手头有股票时的最大利润
```

因此，`dp0[0][j]`对于所有j都要初始化为0，而`dp1[0][j]`对于所有j都要初始化为`-prices[i]`。如果我们将dp0所有值都求出来了，那么很明显`dp0[n-1][k]`就是在最后一天结束时已进行k次交易且手头无股票时的最大收益，也即返回结果。
先看初始状态:
* 当`i==0 && j>=0`: `dp0[0][j] = 0`, `dp1[0][j] = -prices[0]`;
* 当`i>0 && j==0`: `dp0[i][0] = 0`, `dp1[i][0] = max(dp1[i-1][0],  -prices[i])`;

再来考虑状态转移方程，当`i>0`且`j>0`时有
```
dp0[i][j] = max(dp0[i-1][j], dp1[i-1][j-1] + prices[i]) # 保持 or 卖出
dp1[i][j] = max(dp1[i-1][j], dp0[i-1][j] - prices[i]) # 保持 or 买入
```

有了状态定义及转移方程，剩下就好办了。接下来针对具体问题具体分析。


# 2. 具体分析
## 2.1 买卖一次
[121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

此题比较简单，可以不按照上述思考进行分析。要想获益最大，肯定是低价买入高价卖出，所以最简单的方法就是从前向后遍历数组，记录当前出现过的最低价格`cur_min_price`，作为买入价格，并计算以当天价格出售的收益，作为可能的最大收益，整个遍历过程中，出现过的最大收益就是所求。此代码略。


我们看看此题对应1.2节分析的情况，此时只能买卖一次，k = 1，此时代码就为
``` C++
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if(!n) return 0;
    
    vector<vector<int>>dp0(n, vector<int>(2, 0));
    vector<vector<int>>dp1(n, vector<int>(2, 0));
    
    dp1[0][0] = -prices[0];
    for(int i = 1; i < n; i++){
        // j = 0
        dp1[i][0] = max(dp1[i-1][0], -prices[i]);

        // j=1
        dp0[i][1] = max(dp0[i-1][1], dp1[i-1][0] + prices[i]); // 保持 or 卖出
        // dp1[i][1] = max(dp1[i-1][1], dp0[i-1][1] - prices[i]);
        
    }
    return dp0[n-1][1];
}
```
代码中注释部分是一些没有必要的部分，另外空间还可以进行优化，因为dp[i]只与dp[i-1]有关，所以i这一维是没必要的。这是动态规划空间复杂度优化的常用思路，优化后的代码如下：
``` C++
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if(!n) return 0;
    
    vector<int>dp0(2, 0);
    vector<int>dp1(2, 0);
    
    dp1[0] = -prices[0];
    for(int i = 1; i < n; i++){
        int tmp = dp1[0];
        // j = 0
        dp1[0] = max(dp1[0], -prices[i]);

        // j=1
        dp0[1] = max(dp0[1], tmp + prices[i]); // 保持 or 卖出
        
    }
    return dp0[1];
}
```

由于此题比较简单，所以上述代码显得有些繁琐，其实有更加简洁的动态规划方法解此题：定义`dp[i]`代表"在第i天卖出时的最大获益"，则`dp[i]`的值该如何得到？根据在哪一天买入我们可以分成两种情况：
1. 在第i天买入在第i天卖出；
2. 在第i天前某一天买入在第i天卖出；

对于第1种情况，即在同一天买入卖出，获益0；对于第2种情况，因为`dp[i-1]`代表在第i-1天卖出时的最大获益，那如果我在第i-1天不卖而是在第i天卖不就是这种情况下的最大获益吗，此时获益为`dp[i-1] + prices[i] - prices[i-1]`。所以状态转移方程为
```
dp[i] = max(0, dp[i-1] + prices[i] - prices[i-1])
```
在从前往后更新dp时我们还需要用一个变量记录全局的最大获益，时空复杂度均为O(n)。代码如下：
``` C++
int maxProfit(vector<int>& prices) {
    if(prices.size() == 0) return 0;
    vector<int>dp(prices.size(), 0);
        
    int max_profit = 0;
    for(int i = 1; i < prices.size(); i++){
        dp[i] = max(0, prices[i] - prices[i-1] + dp[i-1]);
        max_profit = max(max_profit, dp[i]);
    }
    return max_profit;
}
```

从状态转移方程看出，`dp[i-1]`只与前一天`dp[i-1]`有关，所以也可采取前面类似的思路对空间进行优化，优化后代码如下：
``` C++
int maxProfit(vector<int>& prices) {
    if(prices.size() == 0) return 0;
    int dp = 0;
        
    int max_profit = 0;
    for(int i = 1; i < prices.size(); i++){
        dp = max(0, prices[i] - prices[i-1] + dp);
        max_profit = max(max_profit, dp);
    }
    return max_profit;
}
```

## 2.2 不限买卖次数
[122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

此时不考虑买卖次数，所以我们采用1.2分析的话dp数组就只需要两维:
```
dp0[i]: 第i天结束，手头没有股票时的最大利润
dp1[i]: 第i天结束，手头有股票时的最大利润
```
此时代码如下：
``` C++
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if(!n) return 0;
    
    vector<int>dp0(n, 0);
    vector<int>dp1(n, 0);
    
    dp1[0] = -prices[0];
    for(int i = 1; i < n; i++){             
        dp0[i] = max(dp0[i-1], dp1[i-1] + prices[i]);
        dp1[i] = max(dp1[i-1], dp0[i-1] - prices[i]);
    }
    return dp0[n-1];
}
```
同理，这里空间也可优化，这里就不贴代码了。

类似上一题，此题比较简单，所以采用1.2的思路显得很繁琐，有更加简洁的动归方法：用`dp[i]`代表"在第i天卖出时的最大获益"（那么此时`dp[i]`肯定是单调增数组），状态转移方程为
```
dp[i] = dp[i-1] + max(0, prices[i] - prices[i-1])
```
在更新完毕后，`dp[n-1]`即最终结果，时空复杂度均为O(n)。我们依然可以采用上一题提到的方法将空间复杂度优化到常数，代码如下。
``` C++
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if(!n) return 0;
    int dp =  0;
    
    int max_profit = 0;
    for(int i = 1; i < n; i++)
        dp = dp + max(0, prices[i] - prices[i-1]);
    
    return dp;
}
```

## 2.3 买卖两次
[123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

此题要求最多买买两次，即k=2，按照1.2的分析，我们有代码：
``` C++
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if(!n) return 0;
    
    vector<vector<int>>dp0(n, vector<int>(3,  0));
    vector<vector<int>>dp1(n, vector<int>(3,  0));
    
    dp1[0][0] = -prices[0]; dp1[0][1] = -prices[0]; // i = 0
    for(int i = 1; i < n; i++){
        dp1[i][0] = max(dp1[i-1][0], -prices[i]); // j = 0
        
        // j = 1
        dp0[i][1] = max(dp0[i-1][1], dp1[i-1][0] + prices[i]); // 保持 or 卖出
        dp1[i][1] = max(dp1[i-1][1], dp0[i-1][1] - prices[i]); // 保持 or 买入
        // j = 2
        dp0[i][2] = max(dp0[i-1][2], dp1[i-1][1] + prices[i]); // 保持 or 卖出
        // dp1[i][2] = max(dp0[i-1][2], dp0[i-1][2] - prices[i]); // 保持 or 买入
    }
    return dp0[n-1][2];
}
```
代码中注释部分是一些没有必要的部分，另外空间也可以进行优化，优化后的代码就不贴了。

此题还有一个常见的解法，将在2.4节介绍。


## 2.4 买卖k次
此时就是1.2节分析的一般情况了，代码如下：
``` C++
int maxProfit(int k, vector<int>& prices) {
    int n = prices.size();
    if(!n || !k) return 0;
    
    if (k > n / 2) { // 当k很大时相当于不限制次数
        int res = 0; int hold = prices[0];
        for (int i = 1; i < n; i++)
            res += max(0, prices[i] - prices[i - 1]);
        return res;
    }
    
    vector<vector<int>>dp0(n, vector<int>(k+1,  0));
    vector<vector<int>>dp1(n, vector<int>(k+1,  0));
    
    for(int j = 0; j <= k; j++) dp1[0][j] = -prices[0]; // i = 0
    for(int i = 1; i < n; i++){
        dp1[i][0] = max(dp1[i-1][0], -prices[i]); // j = 0
        
        for(int j = 1; j <= k; j++){ // j > 0
            dp0[i][j] = max(dp0[i-1][j], dp1[i-1][j-1] + prices[i]); // 保持 or 卖出
            dp1[i][j] = max(dp1[i-1][j], dp0[i-1][j] - prices[i]); // 保持 or 买入
        }

    }
    return dp0[n-1][k];
}
```
需要注意的是，LeetCode给了非常极端的测试样例，就是 k 非常大，但是我们稍加思考就知道当`k > n/2` 时就相当于不限制买卖次数了，因为我们最多进行`n/2`次有效的买卖，即第0天买第1天卖、第2天买第3天卖...，所以在上面代码中，我们先判断是否满足`k > n/2`，若是则直接按照不限制次数（2.2节）的思路返回结果。

前面提到，由于当前状态的值只与上一个状态有关，所以我们可以进行空间优化，优化后空间复杂度为O(k)，代码如下：
``` C++
int maxProfit(int k, vector<int>& prices) {
    int n = prices.size();
    // k = min(n / 2, k);
    if(!n || !k) return 0;
    
    if (k > n / 2) { // 当k很大时相当于不限制次数
        int res = 0; int hold = prices[0];
        for (int i = 1; i < n; i++)
            res += max(0, prices[i] - prices[i - 1]);
        return res;
    }
    
    vector<int>dp0(k+1,  0);
    vector<int>dp1(k+1,  0);
    
    for(int j = 0; j <= k; j++) dp1[j] = -prices[0];
    for(int i = 1; i < n; i++){
        dp1[0] = max(dp1[0], -prices[i]); // j = 0
        
        for(int j = 1; j <= k; j++){ // j > 0
            int pre_dp0 = dp0[j]; // 上一次循环的dp0[j]备份
            dp0[j] = max(dp0[j], dp1[j-1] + prices[i]); // 保持 or 卖出
            dp1[j] = max(dp1[j], pre_dp0 - prices[i]); // 保持 or 买入
        }

    }
    return dp0[k];
}
```

此外，此题还有一个常见的解法，也是用两个动归数组：local和global，意义如下：
```
local[i][j]: 已买卖j次且在最后一次是在i天卖出的最大获益；
global[i][j]: 截止到第i天，买卖j次的最大获益；
```
在进行状态计算时只可能有两种情况：
1. 保持现状（或理解成在第i天买入后当天就卖出）
2. 在第i天前某一天买入，在第i天卖出；

所以状态转移方程为
```
local[i][j] = max(global[i-1][j-1], prices[i] - prices[i-1] + local[i-1][j]); // 情况1 or 情况2
global[i][j] = max(global[i-1][j], local[i][j]);
```
所以代码如下（进行了空间优化）：
``` C++
int maxProfit(int k, vector<int>& prices) {
    int n = prices.size();
    if(!n || !k) return 0;
    
    if (k > n / 2) { // 当k很大时相当于不限制次数
        int res = 0; int hold = prices[0];
        for (int i = 1; i < n; i++)
            res += max(0, prices[i] - prices[i - 1]);
        return res;
    }
    
    vector<int>local(k+1, 0);
    vector<int>global(k+1, 0);

    for(int i = 1; i < n; i++){
        for(int j = 1; j <= k; j++){
            local[j] = max(global[j-1], prices[i] - prices[i-1] + local[j]);
            global[j] = max(global[j], local[j]);
        }
    }
    return global[k];
}
```
以上两种解法都很巧妙，注意区别与联系。

## 2.5 带有冷却的股票买卖问题

[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

这道题在买卖不受限制题目的基础上，加了新的限制：卖出后必须至少隔一天才能继续买，也即新增了一个状态："处于冷却期"。此时，"手头有股票"只能从"处于冷却期"转移而来，而不是之前那样从"手头无股票"转移而来。我们需要开辟三个动归数组:
```
dp0[i]: 第i天结束，手头没有股票时的最大利润
dp1[i]: 第i天结束，手头有股票时的最大利润
cool[i]: 第i天结束且第i天处于冷冻期时的最大利润
```
状态转移方程为：
```
dp0[i] = max(dp0[i-1], dp1[i-1] + prices[i]); // 保持 or 卖出
dp1[i] = max(dp1[i-1], cool[i-1] - prices[i]); // 保持 or (冷却 -> 买入)
cool[i] = max(cool[i-1], dp0[i-1]); // 第i-1天结束时没有股票说明第i天可以是冷却期
```
所以有如下代码（进行了空间优化）：
``` C++
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if(!n) return 0;
    
    // vector<int>dp0(n, 0);
    // vector<int>dp1(n, 0);
    // vector<int>cool(n, 0);
    // dp1[0] = - prices[0]; // i = 0
    // for(int i = 1; i < n; i++){
    //     dp0[i] = max(dp0[i-1], dp1[i-1] + prices[i]); 
    //     dp1[i] = max(dp1[i-1], cool[i-1] - prices[i]);
    //     cool[i] = max(cool[i-1], dp0[i-1]);
    // }
    // return dp0[n-1];
    // 空间优化后:
    int dp0 = 0, dp1 = - prices[0], cool = 0;
    for(int i = 1; i < n; i++){
        int pre_dp0 = dp0;
        dp0 = max(dp0, dp1 + prices[i]);
        dp1 = max(dp1, cool - prices[i]);
        cool = max(cool, pre_dp0);
    }
    return dp0;
}
```

## 2.6. 带有交易费用的股票买卖问题

[714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

这道题在买卖不受限制题目的基础上，加了新的条件：每买卖一次需要交一次交易费用。那其实在买卖不受限制题目的基础上，唯一需要改变的就是在卖出（或买入）的时候减去交易费用。代码如下（进行了空间优化）：
``` C++
int maxProfit(vector<int>& prices, int fee) {
    int n = prices.size();
    if(!n) return 0;

    // vector<int>dp0(n, 0);
    // vector<int>dp1(n, 0);
    // dp1[0] = - prices[0]; // i = 0
    // for(int i = 1; i < n; i++){
    //     // 唯一和122不限次数不同就是减去fee
    //     dp0[i] = max(dp0[i-1], dp1[i-1] + prices[i] - fee); 
    //     dp1[i] = max(dp1[i-1], dp0[i-1] - prices[i]);
    // }
    // return dp0[n-1];
    // 空间优化:
    int dp0 = 0, dp1 = - prices[0];
    for(int i = 1; i < n; i++){
        int pre_dp0 = dp0;
        // 唯一和122不限次数不同就是减去fee
        dp0 = max(dp0, dp1 + prices[i] - fee); 
        dp1 = max(dp1, pre_dp0 - prices[i]);
    }
    return dp0;
}
```

# 3 总结
以上六道题就是 LeetCode 当中股票系列的全部内容，此类问题关键就是如何定义状态及状态转移方程，上述设定状态和定义转移方程的思想是可以复用到其他类型的动归问题当中去的，总的来说就是根据有所有不确定的变量（例如这里就是手头是否持有股票等）来定义状态，根据当前状态和之前状态的关系来确定转移方程，这个需要平时的积累和大量的刷题练习。另外上面用到的空间优化方法是动归里很常用的方法，务必掌握。


# 参考文献
[[1] 股票问题汇总.](https://juejin.im/post/5cc501e7e51d456e2d69a7f4)
[[2]【Leetcode 动态规划】 买卖股票 I II III IV 冷却，共5题.](https://blog.csdn.net/Dr_Unknown/article/details/51939121)


------------------
更多我的LeetCode中文题解，可前往GitHub查看：https://github.com/ShusenTang/LeetCode






