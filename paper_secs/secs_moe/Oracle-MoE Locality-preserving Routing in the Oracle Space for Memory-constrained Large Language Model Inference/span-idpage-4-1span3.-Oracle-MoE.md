# <span id="page-4-1"></span>3. Oracle-MoE

Based on the above analysis, we propose Oracle-MoE, replacing the token-level MoE routing mechanism with a better semantic locality in the oracle space and expert activation consistencies for edge-side devices to generate text scenarios.

![](_page_4_Figure_19.jpeg)

<span id="page-4-2"></span>Figure 5. Overview of an Oracle-MoE layer. Residual connections are omitted.

Oracle Space Initialization We obtain the initial oracle space after a short warm-up training phase of the tokenlevel MoE model. After the warm-up stage, we randomly sample N data, and get the semantic group embeddings of each data as is mentioned in Section [2,](#page-1-3) Definition [2,](#page-3-1) and [3.](#page-3-2) These semantic group embeddings form an initial oracle space. Routing in the oracle space does not necessitate complete information; only distinguishing high-level semantics is required. Therefore, to improve computational

efficiency, we adopt SVD to reduce these embeddings to lower dimensions [\(Schmidt, 2020\)](#page-10-13).

Oracle-MoE Routing at Pretraining Since the semantic group embedding varies only locally within a small region in the oracle space as tokens are generated, to ensure tokens in the same semantic group to be dispatched to the same expert, we run K-means [\(Jain, 2008\)](#page-9-10) in the oracle space to get k oracle clusters. The parameter k here is equal to the expert number of the original token-level MoE, and Section [5](#page-5-0) will show that our method yields persistently good results with k ranging from 4 to 32. In the following training process, we replace the routing of the original token-level MoE with the routing mechanism shown in Figure [5.](#page-4-2) For each new incoming data, we first divide it into semantic groups according to the attention scores and get their semantic group embeddings in the oracle space with the same SVD transform matrix computed in the previous stage. Then we calculate which oracle cluster each semantic group belongs to as Equation [1,](#page-4-3) and dispatch tokens in the semantic group to the corresponding expert.

Oracle-MoE Routing at Inference The inference stage can be divided into prefill and decode stages. The prefill stage is the same as the training stage routing. In the decode stage, Oracle-MoE first decides which semantic group the coming token belongs to, and updates the semantic group embedding with the coming token. Since KV cache is a widely adopted strategy [\(Yuan et al., 2024\)](#page-10-14), this does not introduce memory overhead. Then we dispatch it to the expert corresponding to the oracle cluster of its semantic group. In our experiments, there are often fewer than 5 semantic groups in an input session within a length of 1024, and semantic groups from the same session are likely to belong to the same oracle cluster. So, the oracle space routing preserves the semantic locality of input tokens and yields a low expert variation, contributing to low expert swapping latency.

