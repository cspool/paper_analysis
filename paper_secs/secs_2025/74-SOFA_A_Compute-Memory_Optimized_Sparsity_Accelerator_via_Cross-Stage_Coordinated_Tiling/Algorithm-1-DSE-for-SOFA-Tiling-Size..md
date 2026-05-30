# Algorithm 1: DSE for SOFA Tiling Size.

```
1 Input: Evaluation function \mathcal{L} and exploration space \Theta;

2 Initial: Max Iter T, sample \mathcal{D}_t = \{R_i, \mathcal{L}(R_i), i=1,...,n\},

Best target function result \mathcal{J} = \infty;

3 while t < T and result does not converge do

4 |R_t \leftarrow \arg \max_{\Theta} \alpha(\Theta, \mathcal{D}_t), \mathcal{J}_{new} \leftarrow \mathcal{L}(R_t);

5 |\mathcal{D}_{t+1} \leftarrow \{\mathcal{D}_t, (R_t, J_{new})\};

6 |\mathcal{GP}_{new} \leftarrow \text{Update}(\mathcal{GP}, \mathcal{D}_{t+1});

7 |\text{if } \mathcal{J}_{new} < \mathcal{J} \text{ then}

8 |\mathcal{J} \leftarrow \text{Update}(J_{new});
```

![](_page_6_Figure_8.jpeg)

Fig. 11. High-level block diagram for the SOFA accelerator.

5% - 50%, step=5%, to ensure that we can obtain a high-quality solution. However, such space is huge and unaffordable for brute force search. Taking BERT-Base with 12 Transformer layers as an example, we need to search for the optimal choice in a 26-dimensional space consisting over  $10^{15}$  choices. Even though the inference on highly parallel GPU clusters costs less than 1 ms, it will take unbearable time consumption (over  $10^8$  h) using traversal-based grid search for this remarkable design space. To this end, we apply a Bayesian optimization method to execute the search process. The targeted optimization problem (modeled as a Gaussian Process (GP) in Bayesian optimization) is constructed concerning both the accuracy and the computational complexity, which is formalized as Eq. (2).

minimize 
$$\mathcal{L}(R) = \mathcal{L}_{en} + \alpha \times \mathcal{L}_{cmp} + \beta \times \mathcal{L}_{exp},$$
 (2)

$$\mathcal{L}_{cmp} = \sum_{i} (B_{ci} \cdot k) / \sum_{i} (S \cdot k), \tag{3}$$

$$\mathcal{L}_{exp} = \sum_{i} (S/B_{ci}), \tag{4}$$

where R is the hyperparameter vector composed of the k and  $B_{ci}$  of each layer,  $\mathcal{L}_{en}$  is the cross-entropy loss,  $\mathcal{L}_{cmp}$  and  $\mathcal{L}_{exp}$  are the penalty terms for computation overhead, as defined in Eqs. (3) and (4).  $\alpha$  and  $\beta$  are two coefficients to balance the accuracy and performance. The whole searching process is summarized in Alg. 1.

#### IV. ARCHITECTURE AND HARDWARE INNOVATION

Despite substantial algorithmic acceleration, a naive implementation of SOFA faces three challenges. First, LP is crucial in predicting vital tokens. It must ensure high precision

![](_page_7_Figure_0.jpeg)

Fig. 12. Architecture for the cross-stage DLZS prediction.

and low power consumption. Additionally, the top-k engine must support variable-length inputs and high throughput within low power overhead, due to the flexible tiling execution and high parallelism of LTPP. Second, specific architecture and datapath designs are needed to support the intra-stage operator-fusion paradigm of SU-FA for enhanced efficiency. Finally, during LTPP execution, the varying requirements of K and V for each query may lead to redundant memory access, thus necessitating a memory-efficient scheduling strategy.

#### A. Architecture Overview

Fig. 11 depicts SOFA's overall architecture, which comprises six main modules: on-chip SRAM storage, a DLZS prediction unit, an iterative SADS unit, a PE array, an SU-FA unit, and a tiled & out-of-order controller. SOFA is designed to process 128 queries in parallel. First, the indices of tokens and corresponding  $W_k$  of a tile produced by the *controller* are sent to data fetcher, which calculates the physical address and fetches data to on-chip SRAM **①**. Then, the *DLZS predictor* starts to estimate matrices K and A with log-based shift and summations **2**. Next, the 128-row **A** is sent to SADS unit, to find out the top-k important Q-K pairs  $\odot$ . Subsequently, the sorting results are sent back to the controller **4**, which generates a top-k mask, then data fetcher reads corresponding data according to the mask **6**. After that, the scheduler controls the *PE array* to generate the necessary Ks and Vs **6**. Later, the generated KVs are sent to the SU-FA unit to execute computememory efficient attention calculations **7**. Finally, the outputs of attention are stored to off-chip DRAM 3.

