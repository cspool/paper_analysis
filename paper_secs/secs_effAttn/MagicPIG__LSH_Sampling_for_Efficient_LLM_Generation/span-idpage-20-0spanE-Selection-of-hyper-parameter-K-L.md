# <span id="page-20-0"></span>E Selection of hyper-parameter (K, L)

In this section, we discuss the impact of the LSH hyper-parameter (K, L) and how to select it. First, we briefly explain what hyper-parameter (K, L) does for LSH sampling. Then, we explain the relations between (K, L) and attention computation cost and accuracy. Finally, we show how we decide the parameters by ablation studies.

## E.1 (K, L) in LSH

In each hash table, we use K hash functions to compute the hash code of k and q. In Simhash [\(Charikar,](#page-12-11) [2002\)](#page-12-11), the hashing we use in MagicPIG, the hash functions are random projections. With K random projections, we are able to partition the space (in our problem, the space is R128) into 2<sup>K</sup> subspace. If and only if k and q fall in the same subspace, we say they collide in this hash table. We have L hash tables in total. In MagicPIG, if and only if k and q collide in at least two hash tables, k is sampled by q. Here are some intuitions about how (K, L) will influence the LSH sampling in MagicPIG.

- If K is too small, then we cannot partition the space well; we will sample too many ks, which might be far away from q (in the attention problem, this means their inner production is small), increasing computation cost.
- On the other hand, if K is too large, although the quality of sampled ks will be better, the collision probability in each table will be small; thus, the number of the sampled ks will be reduced. We need to increase L to ensure that a certain number of keys are sampled and involved in the computation. However, increasing (K, L) too much will bring more memory overhead on CPU DRAM since we build L hash tables for each key-value head.

Thus, (K, L) is important because it balances computation cost, overhead, and sampling quality (which determines accuracy). Tuning (K, L) is necessary in LSH [\(Lv et al.,](#page-14-17) [2017;](#page-14-17) [Slaney et al.,](#page-15-15) [2012\)](#page-15-15).

