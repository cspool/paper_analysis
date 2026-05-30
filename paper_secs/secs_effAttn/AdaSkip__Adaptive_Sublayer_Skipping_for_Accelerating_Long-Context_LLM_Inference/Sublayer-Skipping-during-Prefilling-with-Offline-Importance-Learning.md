# Sublayer Skipping during Prefilling with Offline Importance Learning

It is necessary to skip layers in prefilling phases during long-context inference, since the prefilling phase results in unacceptably high TTFT and substantial KV cache demands. However, existing layer-skipping strategies rarely consider skipping strategies in such phases. Moreover, since different models exhibit various similarity distributions, current fixed layer-skipping strategies cannot achieve optimal results. The primary obstacle in devising an adaptive sublayer-wise skipping approach for the prefilling phase lies in the absence of prior knowledge before execution. To address this challenge, we propose an offline importance learning method that leverages the high correlation between historical prefilling features and new prefilling features.

**Insight.** Using sublayer-wise IO similarity feature from historical tasks can precisely predict the sublayer-wise skipping behavior for prefilling new inference tasks. We perform the IO similarity analysis study of running inference tasks of

![](_page_3_Figure_0.jpeg)

Figure 3: IO similarities of attention (ATTN) and FFN modules in different layers.

![](_page_3_Figure_2.jpeg)

Figure 4: IO similarities of sublayer modules in prefilling (P) and decoding (D) phases.

| Type | Src      | Dest     | Layer Hit Rate          |  |  |  |
|------|----------|----------|-------------------------|--|--|--|
| ATTN | TriviaQA | MFieldQA | 3.76/4, 4.86/6, 9.31/10 |  |  |  |
| ATTN | MFieldQA | Wiki     | 3.80/4, 5.54/6, 9.90/10 |  |  |  |
| ATTN | TriviaQA | Wiki     | 3.79/4, 5.50/6, 9.68/10 |  |  |  |
| FFN  | TriviaQA | MFieldQA | 3.66/4, 5.69/6, 9.56/10 |  |  |  |
| FFN  | MFieldQA | Wiki     | 3.77/4, 5.97/6, 9.38/10 |  |  |  |
| FFN  | TriviaQA | Wiki     | 3.75/4, 5.96/6, 9.64/10 |  |  |  |

Table 1: Average hit rate of unimportant layers using historical features across datasets in prefilling phases.

multiple datasets (Taori et al. 2023) including 2WikiMQA, MultiFieldQA-en, and TriviaQA using LLaMA3.1-8B-128k and quantify the average hit rate of unimportant layers in the prefilling phase. We record the average IO similarity on the Src dataset in prefilling phases and test the hit rate on the Dest dataset. The results shown in Table 1 reveal that historical IO similarity in prefilling phases gains a high hit rate for subsequent tasks, suggesting that this feature can be used in prediction and shared across different datasets.

**Method.** Based on the insight, the major workflow of offline importance learning consists of the similarity study and the corresponding deviation correction procedure. Specifically, suppose N inference tasks (samples) are used in offline importance learning. As for the inference task  $T_i$  with prompt length  $|T_i|$ . Suppose that the model has M transformer layers with M attention sublayers and M FFN sublayers. We first

take notes of average similarity  $Simi\bar{l}arity$  in the prefilling phase. The average similarity of the  $j_{th}$  sublayer,  $Simi\bar{l}arity_j$ , can be accumulated as:

$$\textit{Similarity}_{j} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{|T_{i}|} \textit{Similarity}(\vec{a}_{it}^{j}, \vec{b}_{it}^{j})}{\sum_{i=1}^{N} |T_{i}|} \tag{2}$$

where  $\vec{a}_{it}^j$  and  $\vec{b}_{it}^j$  are the input and output vectors of the t-th token in task i. In addition, if the angle between vector  $\vec{a}_{it}^j$  and  $\vec{b}_{it}^j$  is not very large, the proportion of modulus of  $\vec{a}_{it}^j$  and  $\vec{b}_{it}^j$  relatively become prominent, suggesting some compensation needs to be applied.

However, due to the residual connections employed between each sublayer, the modulus of the input and output of one layer has minor variations, which implies that the average proportion of modulus can effectively compensate for the deviations. Hence, we use the average proportion of historical modulus of  $\vec{a}_{it}^j$  and  $\vec{b}_{it}^j$  in  $j_{th}$  layer to scale  $\vec{a}_{it}^j$  so that output vector  $\vec{b}_{it}^j$  is close to original  $\vec{b}_{it}^j$ . The average scale factor of j-th sublayer,  $Scale_j$ , can be formulated as:

$$Scale_{j} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{|T_{i}|} \frac{\|\vec{b}_{it}^{j}\|}{\|\vec{a}_{it}^{j}\|}}{\sum_{i=1}^{N} |T_{i}|}$$
(3)

we use  $S\bar{cale}_j$  to compensate the input  $\vec{a}_{it}^j$ , getting approximate output:

$$\vec{b}_{it}^{\hat{j}} = Scale_i * \vec{a}_{it}^j \tag{4}$$

| Dataset   | Size | Layer Hit Rate         |
|-----------|------|------------------------|
| TREC      | 5    | 0.84/2, 2.67/4, 4.70/6 |
| TREC      | 20   | 1.08/2, 3.04/4, 4.90/6 |
| TREC      | 40   | 1.07/2, 3.09/4, 4.90/6 |
| GovReport | 5    | 1.01/2, 2.94/4, 4.97/6 |
| GovReport | 20   | 1.14/2, 3.01/4, 5.02/6 |
| GovReport | 40   | 1.19/2, 3.03/4, 5.03/6 |
|           |      |                        |

Table 2: Average hit rate of unimportant layers identified through different window sizes in the decoding phase.

After obtaining *Similarity* and *Scale* of each sublayer module, we sort all sublayers in descending order based on their *Similarity*, getting the sorted list *sorted* with 2M elements. Since there is a trade-off between the number of skipped layers and the generation quality, we introduce an acceleration ratio,  $\alpha$ , as a knob to control this trade-off. Given the acceleration ratio  $\alpha$ , the number of sublayers to be skipped, m, can be calculated as  $m=M-\frac{M}{\alpha}$ , and the targeted skipping sublayer number is 2m. The top 2m sublayers in the *sorted* list are selected, forming the *skipped* set.

