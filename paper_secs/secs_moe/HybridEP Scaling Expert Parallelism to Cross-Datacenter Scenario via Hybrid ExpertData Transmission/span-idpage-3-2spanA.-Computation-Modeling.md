# <span id="page-3-2"></span>A. Computation Modeling

**GeMM Modeling.** The computation latency mainly comes from General Matrix Multiplication (GeMM) operations. Following prior works [20], [32], we use a linear model to estimate the latency. Given two matrices to be multiplied, with size (L,H) and (H,M) respectively, the latency of a single GeMM operation can be expressed as:

$$Lat_{comp}^{GeMM} = \frac{LMH}{C},\tag{1}$$

![](_page_3_Picture_10.jpeg)

Fig. 5. The modeling of training process and the communication breakdown of A2A and AG. (a) A shows a modeling process using the divide-and-conquer approach. The training process is first split into independent modeling of computing and communication streams, and then their overlapping relationships are considered for merging. (b) shows that the traffic of A2A remains unchanged (i.e., O(1)), while the traffic of AG is multiplied by number of GPUs (i.e., O(n)).

<span id="page-3-1"></span>where C represents the average computation throughput of GPU. Note C will be reduced if GeMM is too small to utilize GPU power. However, this will not affect overall modeling effectiveness due to its small overhead, confirmed by [20].

Computation Stream Modeling. Assume that there are m transformer blocks before a MoE block, The computation latency can be expressed as

<span id="page-3-5"></span>
$$\begin{split} Lat_{comp} &= mLat_{comp}^{TF} + Lat_{comp}^{MoE} \\ &= (m+1)Lat_{comp}^{Att} + mLat_{comp}^{FFN} + nLat_{comp}^{Ep}, \ \ (2) \end{split}$$

where TF, MoE represents the transformer and MoE block. Att, FFN, Ep represents the computation process of attention, FFN, and expert, which consists of multiple GeMM operations. Here we can consider their latency as constant. n is the number of experts on one GPU, thus expert computation latency is repeated by n times. For brevity, we consider  $(m+1)Lat_{comp}^{Att}+mLat_{comp}^{FFN}$  as Pre-Expert, denoted  $Lat_{comp}^{PE}$ .

