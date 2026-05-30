# <span id="page-4-1"></span>**Algorithm 1** Similar Prompts Searching (SPS)

```
Input: \alpha
1: Initialize clustering tree tree
   while new prompt prom arrives do
2:
3:
       \mathbb{PROM} = \Pi
       Select one leaf node leaf in tree
4.
5:
       Put samples from leaf into PROM
6:
       if len(\mathbb{PROM}) < \alpha then
          Turn to leaf's siblings and update leaf
7:
          Add samples into PROM until \alpha samples are obtained
8:
9.
       end if
       return PROM
10:
11: end while
```

SPS algorithm is outlined in Algorithm 1, where  $\mathbb{PROM}$  represents the set of top- $\alpha$  similar historical prompts searched currently. SPS initially builds the clustering tree tree (Line 1). For each new prompt, a leaf leaf is identified to retrieve similar prompts (Lines 2-5). If insufficient samples exist in leaf, its siblings are turned to (Lines 6-9). After acquiring  $\alpha$  historical prompts, the set  $\mathbb{PROM}$  is returned (Lines 10-11).

**Expert Activation Distribution Prediction.** For each historical prompt  $\zeta_j$ , we obtain its expert activation distribution matrix  $\tilde{S}$ . Matrix element  $\tilde{s}_{l,k} = \frac{frec_{l,k}}{\sum_k frec_{l,k}}$  represents the "linear scaling activation frequency" of expert  $e_{l,k}$  during prefilling of  $\zeta_j$ .  $frec_{l,k}$  is the times  $e_{l,k}$  is activated.  $\sum_k frec_{l,k}$  equals product of the number of  $\zeta_j$ 's tokens  $N_j^{in}$  and the number of experts activated by one token in each layer  $N^{topk}$ .

SCS between the new prompt and the retrieved  $\alpha$  historical prompts are converted into probability weights via soft-

max. The expert activation distribution matrices of historical prompts are then weighted-summed to predict the result.

