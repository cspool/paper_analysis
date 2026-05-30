# <span id="page-4-1"></span>5 Scheduling Algorithms

The offline scheduler of HYTE aims to determine a relatively optimized initial tiling scheme for the given sparse data. This scheme should include the parameters in Figure 2. Similar to existing schedulers for dense computations [13, 16, 28, 39], the scheduler searches the configuration space of different tile shapes, inter-tile orders, etc., assesses each scheme with a hardware cost model, and identifies the best scheme with the minimum cost (i.e., the best performance).

## **Algorithm 1:** Overall workflow of HYTE scheduler.

```
// Sample and estimate

1 S_I \leftarrow Sample a fraction sp of values from 0 to I;

2 S_J \leftarrow Sample a fraction sp of values from 0 to J;

3 effMAC ← ESTEFFMAC(A, B, S_I, S_J, \text{sp});

4 {nnzCTk}_{T_k=1,2,4,...,K} \leftarrow ESTNNzCTK(A, B, S_I, S_J, \text{sp}, \text{sk});

// Search the tiling scheme with the minimum cost

5 c_{\min} = \infty; s_{\min} = \bot;

6 foreach s in Pruned Tiling Scheme Space() do

7 c \leftarrow CostModel(s, effMAC, nnzCTk_{T_k});

8 c \leftarrow if c < c_{\min} then c_{\min} \leftarrow c; c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin c \in sin
```

<span id="page-4-6"></span><span id="page-4-5"></span>The overall workflow is summarized in Algorithm 1. The key challenge that the scheduler needs to address is that, unlike the dense case where the computation load and data sizes are all directly known, in the sparse scenario both the required computation

amount and the output tensor size would depend on the input tensors' data distributions as well as their correlation in a complex way. HYTE proposes *effective yet lightweight sampling and estimation methods* (Algorithm 1 Lines 1 to 4; Section 5.1) to derive these required statistics without the need to investigate the full input tensors. We find that such sampling and estimation may introduce 15% on average and up to 43% errors against the cycle-accurate simulation results, mostly occurring in irregular tensors. These inaccuracies can be fixed by the online dynamic fine-tuning phase.

To efficiently explore the search space and reduce the search cost, HYTE leverages several observations to *prune* unnecessary and sub-optimal schemes (Line 6; Section 5.2). The remaining ones are then fed to the hardware cost model (Line 7; Section 5.3) for assessment to determine the best one. We provide several example cases in Section 5.4 to show how the scheduler works.

