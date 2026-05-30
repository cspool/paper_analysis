# **Algorithm 1:** Adaptive Pipeline Granularity Search

```
Input: the batch size of tokens B
   Output: the number of partitions n
1 global: S = \{\};
   global: cache\_table = \{\};
   if B in cache_table then
       return cache\_table[B];
   end
   (\mathcal{R}_n, n) = find(\mathcal{S}, B);
   if n == -1 then
       n = searchBestGran(B);
       (\mathcal{R}_n, n) = find(\mathcal{S}, B);
       if \mathcal{R}_n == \varnothing then
10
           \mathcal{R}_n = range(B, B);
11
           insert(\mathcal{S}, (\mathcal{R}_n, n));
12
13
       else
14
             range(min(B, B_n^{lower}), max(B, B_n^{upper}));
15
   end
16
17 cache\_table[B] = n;
18 return n;
```

#### C. Adaptive Pipelining Granularity Configuration

The effectiveness of pipeline parallelism is largely determined by the granularity of pipeline, which is determined by the number of partitions n. A coarse-grained granularity fails to take the benefit of pipeline because S, C, and R cannot be fully overlapped. On the other hand, a very fine-grained granularity could lead to GPU under-utilization. Therefore, it is necessary to configure for the optimal n at runtime to take full advantage of pipeline parallelism.

We consider the training process of MoE models, in which the batch size of tokens is split into n partitions. The microbatch size equals B/n. It requires running dozens of iterations to search for the optimal configuration of n by calling the method searchBestGran(B). Although the cost can be amortized by epochs, unfortunately, B is dynamic and span a wide range in MoE training [32]. Thus, it is time consuming to search for the optimal n for every value of B.

In order to reduce the searching space, we propose Algorithm 1 based on an intuitive hypothesis: n is monotonically increasing as B increases. As a result, the whole value domain of B can be a set of disjoint ranges  $\{\mathcal{R}_n\}(\mathcal{R}_n = range(B_n^{lower}, B_n^{upper}))$ , which is a one-to-one mapping to n. We denote the set of pairs  $(n, \mathcal{R}_n)$  as S. Given the batch size of tokens B, the optimal n can be looked up by finding a pair  $(n, \mathcal{R}_n)$  that satisfies  $B \in \mathcal{R}_n$  in S, which is shown in line 6. If not found, searchBestGran(B) is called to search

for the optimal configuration n by trials, i.e., lines 7-8. If n is not in S, a new pair  $(n, \mathcal{R}_n = (B, B))$  is inserted into  $\mathcal{R}_s$ , i.e., lines 9-12. Otherwise, we merge B into range  $\mathcal{R}_n$ , as shown in lines 13-14. To eliminate the overhead of find(B) method, we build a hash table to cache the best strategy for each specific B, which is illustrated in lines 3-5. We implement the set  $\mathcal{R}s$ based on the binary-search-tree algorithm. The complexity of find(B) and insert(n, B) are both O(log(n)).

