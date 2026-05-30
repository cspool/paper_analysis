# E.3 (K, L) and computation cost/budget

In summary, increasing K will make the budget[5](#page-21-1) smaller, and increasing L will increase the budget.

Theoretically, as introduced in Section [4.3,](#page-6-1) in our approach, the key k<sup>i</sup> is sampled only if at least two hash tables exist where k<sup>i</sup> shares the hash value with query q. With the assumption that k<sup>i</sup> is well-distributed (In

<span id="page-21-1"></span><sup>5</sup>Cost<sup>2</sup> in Tables [1](#page-9-0) to [3](#page-10-1)

each hash table out of L, each hash value corresponds to roughly the same number of kis), the ratio of retrieved kis can be estimated with

$$\mathcal{B}/n = 1 - (1 - 0.5^K)^L - L \times 0.5^K (1 - 0.5^K)^{L-1}$$
(21)

where n is the context length. Here, we estimate the collision probability of k<sup>i</sup> and q in a single hash table as 0.5 K.

Empirically, the ratio of retrieved keys and values (B/n) might differ from the above estimation since the data is not perfectly distributed. We present the empirically measured budget in Table [9.](#page-22-0)

<span id="page-22-0"></span>Table 9 Empirical measured budget/cost for different (K, L).

| K / L | 75    | 100  | 120  | 150  | 200  | 300   |
|-------|-------|------|------|------|------|-------|
| 7     | 14%   | 21%  | 27%  | 35%  | 48%  | 66%   |
| 8     | 5%    | 8%   | 11%  | 15%  | 22%  | 36%   |
| 9     | 1.6%  | 2.7% | 4%   | 5.4% | 8.5% | 15.4% |
| 10    | 0.5%  | 0.9% | 1.5% | 2%   | 3%   | 6%    |
| 11    | 0.15% | 0.3% | 0.5% | 0.6% | 1%   | 2%    |

