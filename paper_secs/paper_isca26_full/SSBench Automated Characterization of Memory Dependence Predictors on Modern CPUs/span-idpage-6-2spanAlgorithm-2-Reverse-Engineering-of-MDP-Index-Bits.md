# <span id="page-6-2"></span>Algorithm 2 Reverse-Engineering of MDP Index Bits

```
Input: Entry y with size-k minimal eviction set \mathbb{E} = \{x_0, x_1, ..., x_{k-1}\}
     Output: Tag/offset-bit set \mathbb{T} and index-bit set \mathbb{I}
    n \leftarrow \text{qet\_test\_size}()
2: \ \mathbb{S} \leftarrow \{\}
3: h \leftarrow \text{hash}(y)
4: for i = 1 to n do
5:
         Find a new x_t so that \mathbb{E} - \{x_0\} + \{x_t\} can evict y. Add it to set \mathbb{S}.
6: end for
    for each bit b in h do
7:
          if (hash(x) >> b \& 1) is the same for all x \in \mathbb{E} \cup \mathbb{S} then
8:
9:
              Add b to \mathbb{I}.
10:
11:
               Add b to \mathbb{T}.
12:
          end if
13: end for
```

eviction set (i.e., the minimum number of Pairs required to evict an entry from the MDP table). We first train the MDP using  $Pair_{x_0}^y$  by executing  $D_P_{x_0}^y$  repeatedly. Then, for  $Pair_{x_1}^y$ ,  $Pair_{x_2}^y$ , ...,  $Pair_{x_n}^y$  with distinct hash values, we sequentially execute  $D_P_{x_1}^y$ ,  $D_P_{x_2}^y$ , ...,  $D_P_{x_n}^y$  to prime more entries to the MDP table. Finally, we execute  $N_P_{x_0}^y$  and measure the execution time. If we observe  $T(N_P_{x_0}^y) = S$ , it indicates that the training state for  $Pair_{x_0}^y$  has been evicted by other n store-load pairs. The smallest n that causes eviction is the associativity of the MDP table.

Solution to Challenge 3: Avoiding global disabling of the MDP. On some CPUs, triggering too many mispredictions may globally disable the MDP, forcing all loads to stall. To avoid this, we increase the candidate size in powers of two. When we identify an integer k such that  $2^{k-1}$  entries do not cause eviction but  $2^k$  entries do, we perform a linear search between  $2^{k-1}$  and  $2^k$ . This allows us to find the accurate eviction-set size without triggering too many mispredictions.

**MDP prediction table structure characterization.** After constructing the eviction set, we characterize the MDP table size and the number of sets based on the store-load bounce. As shown in Algorithm 2, given an eviction set  $\mathbb E$  for address  $x_0$ , we expand it into a larger set  $\mathbb S$  where each address in  $\mathbb S$  can evict  $x_0$ , and all addresses in  $\mathbb S$  map to the same set in the MDP table. We then analyze the hash-bit positions that remain identical across all addresses in  $\mathbb S$ . These bit positions correspond to the index bits, while the remaining bits are used as tag or offset bits. A larger  $\mathbb S$  yields more accurate results.

Replacement policy characterization. We extend Algorithm 2 to determine the MDP table replacement policy. We generate a minimal eviction set plus four entries (assuming associativity > 4) to test the replacement priority of the first four entries inserted into the MDP table. Following prior work [1], permutation  $\Pi_i$  denotes eviction priority when entries  $x_3$  to  $x_0$  are inserted sequentially and then  $x_i$  is accessed. Table I shows  $\Pi$  values of FIFO, LRU, Tree-PLRU and NLRU.

To evaluate  $\Pi_i$ , we first train the MDP by inserting entries  $x_3$ ,  $x_2$ ,  $x_1$ ,  $x_0$  in order, and then re-access it with  $x_i$ . We use the extended eviction-set and automatically probe which of the four entries remains, thereby determining the replacement priority. We compare the measured  $\Pi_i$  pattern against the

TABLE I REPLACEMENT POLICY FOR FOUR ENTRIES IN FOUR MODELS

<span id="page-7-1"></span>

| Replacement<br>Policy | Π0           | Π1           | Π2           | Π3           |
|-----------------------|--------------|--------------|--------------|--------------|
| FIFO                  | (0, 1, 2, 3) | (0, 1, 2, 3) | (0, 1, 2, 3) | (0, 1, 2, 3) |
| LRU                   | (0, 1, 2, 3) | (1, 0, 2, 3) | (2, 0, 1, 3) | (3, 0, 1, 2) |
| Tree-PLRU             | (0, 1, 2, 3) | (1, 0, 3, 2) | (2, 1, 0, 3) | (3, 0, 1, 2) |
| NLRU                  | (0, 1, 2, 3) | (1, 2, 3, 0) | (2, 0, 3, 1) | (3, 0, 1, 2) |

known replacement policies in Table [I](#page-7-1) to identify the MDP's replacement strategy. Increasing the number of tested entries can improve accuracy, but also introduces more noise. In practice, probing with four entries is sufficient to reliably characterize the MDP's replacement policy.

#### <span id="page-7-0"></span>*F. Automated Security Test*

Once the state machine and the organization of an MDP are characterized, testing its security properties becomes straightforward. For example, to determine whether an MDP is shared across security domains, we simply construct P airs with the hash collision in two security domains [\[45\]](#page-14-25), and alternate the training and probing procedures. To test interactions between the MDP and other microarchitectural mechanisms, such as whether speculative execution or cache misses affect MDP updates, we ensure that the trained P air and the probed P air in the same process have the hash collision.

We present several examples of security tests. First, to test whether an MDP is isolated between two security domains, we place P airs in different cores, processes, or privilege levels. Second, to test whether an MDP updates during speculative execution, we follow the setup in [\[34\]](#page-13-32): we trigger an outof-order execution after an exception, ensuring a P air is executed but not committed. Then we probe the MDP in the exception handler to observe whether an update was triggered by this P air. Finally, we test several update conditions: (1) Chaining: whether the delay of a load blocked by a previous MDP prediction causes MDP updates in subsequent store–load pairs. (2) No Delay: whether a store without delay updates the MDP. (3) Single load: whether a load without a preceding store updates the MDP. It is worth noting that some prior works [\[40\]](#page-14-26), [\[44\]](#page-14-27) suggest that instruction and address attribute randomization can further uncover additional security issues. We leave this extension to future work.

