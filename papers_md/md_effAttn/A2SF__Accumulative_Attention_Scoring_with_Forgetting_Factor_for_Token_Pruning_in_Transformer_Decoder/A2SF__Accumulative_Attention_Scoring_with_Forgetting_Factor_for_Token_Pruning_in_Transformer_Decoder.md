# A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

### Hyun-rae Jo

Department of Electrical and Computer Engineering Sungkyunkwan University smp9898@skku.edu

#### Dongkun Shin

Department of Electrical and Computer Engineering Sungkyunkwan University dongkun@skku.edu

# Abstract

Recently, large language models (LLM) based on transformers are facing memory bottleneck issues due to KV cache, especially in long sequence handling. Previous researches proposed KV cache compression techniques that identify insignificant tokens based on Accumulative Attention Scores and removes their items from KV cache, noting that only few tokens play an important role in attention operations. However, we have observed that the existing Accumulative Attention Score is not suitable for the transformer decoder structure. In the decoder model, the number of times the Attention Score accumulates varies depending on the order of token appearance due to the effect of masking, causing an uneven comparison between tokens. To solve this, we propose Accumulative Attention Score with Forgetting Factor (A2SF) technique, which introduces a "Forgetting Factor" in the Attention Score accumulation process. A2SF applies a penalty to the past Attention Score generated from old tokens by repeatedly multiplying the Forgetting Factor to the Attention Score over time. Therefore, older tokens receive a larger penalty, providing fairness among different ages of tokens. Through the fair comparison among tokens, we can more effectively select important tokens. We have verified the accuracy improvement through A2SF in the OPT and LLaMA models and A2SF improves the accuracy of LLaMA 2 by up to 7.8% and 5.1% on 1-shot and 0-shot. The code is available at <https://github.com/Dirac-Notation/A2SF>

# 1 Introduction

In recent years, Transformers-based Large Language Models (LLMs) [\[1\]](#page-9-0) have made significant strides. These models are now widely used in various domains, including content creation, text summarization, and chatbots. However, the computational and memory demands of these models are escalating rapidly to maintain high accuracy. Unlike the Transformer Encoder that processes the entire input simultaneously, the Transformer Decoder, which feeds the generated token back as input, needs to recalculate the Key and Value for the previously generated tokens. To address the issue posed by this auto-regressive characteristic, a Key-Value (KV) Cache is employed to store the Key and Value of the generated token in memory. The KV Cache, which retains the Key and Value for all tokens, has a memory usage that increases linearly with the sequence length. Consequently, the memory usage tend to escalate with longer sequences and larger batches. In particular, repeated memory access to load the KV Cache can create a bottleneck due to memory bandwidth limitations.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: A2S results of encoder model and decoder model and A2SF results in a winogrande data example. SpAtten and H2O use A2S. Unlike the results in BERT, the comparison of tokens in LLaMA is not being properly conducted. The imbalance can be resolved through A2SF.

In the worst-case scenario, computation may become impossible due to insufficient GPU memory. To mitigate the issue of limited GPU memory capacity, LLM optimization techniques [\[2](#page-9-1)[–4\]](#page-9-2) have proposed offloading the KV Cache to CPU memory. However, this increases data traffic between the GPU and CPU, exacerbating the bottleneck, and as the sequence lengthens, the continually expanding KV Cache can eventually exhaust CPU memory space. Therefore, a fundamental solution is required to effectively reduce the size of the KV Cache.

SpAtten [\[5\]](#page-9-3) illustrated that important tokens can be distinguished through the Accumulative Attention Score (A2S), which accumulates the output values of the Softmax function on a per-token basis, in Encoder models. They proposed a token pruning technique that calculates the A2S of each token while calculating a layer, removes tokens with small A2S during the execution of the next layer, and only uses the remaining tokens. This technique has the advantage of being applicable to the Pre-trained Model in a Plug-and-Play manner without incurring additional costs.

H2O [\[6\]](#page-9-4) extended SpAtten's Token Pruning technique to the Transformer Decoder-based LLM model. Due to the structural differences between the Encoder model and the Decoder model, which has autoregressive characteristics, the method of accumulating A2S across layers was modified to accumulate along the Generative step. By maintaining each A2S in each Attention-head, it was confirmed that the Token Pruning method based on A2S is also effective in LLMs. Subsequent research [\[7–](#page-9-5)[9\]](#page-9-6) improving H2O's technique highlighted problems arising from H2O's token processing and Softmax, and proposed techniques to address them. Consequently, numerous studies are being conducted using A2S as a token selection method, which can effectively distinguish important tokens through a simple method.

Meanwhile, the Transformer Decoder comprises Masked Self-Attention, which applies a Causal Mask during the Attention operation. It masks the upper triangular area of Attention. Through this, the result of the softmax function has a probability of 0 in the masked area. This blocks the connection between the current token and the future token, indicating that the information of the future token cannot be known in the process of generating current tokens.

Despite numerous studies using A2S, they do not consider the characteristics of the above Causal Mask. The output of Masked Self-Attention always has an upper triangular area of 0, leading to an imbalance in the process of A2S depending on the order of token appearance. Tokens that were generated in the past have a lot of A2S because there are few tokens that are masked in the attention operation, and tokens that were generated recently have a small amount of accumulation because the Attention Score is 0 for all tokens before the token. Therefore, the likelihood of measuring the importance of tokens that appeared in the past is high. This problem makes it impossible to make a precise comparison about the tokens, and often selects unnecessary tokens that were generated earlier. This is a critical problem that can lower the accuracy of the model by removing important tokens.

To address this issue, we propose A2SF (Accumulative Attention Score with Forgetting Factor), which can resolve the unfairness between tokens about Accumulative Attention Score. A2SF applies

a Forgetting Factor to the Attention Score to reduce the value of past Attention Scores. By increasing the number of applications of the factor over time, it gradually forgets the past and reduces its impact on the present. Therefore, since many Factors are applied to the Score of old tokens, it can resolve the imbalance with recent tokens.

We conducted experiments with various models such as LLaMA and OPT, and confirmed that A2SF shows accuracy improvement in many datasets compared to existing techniques. We were able to confirm that the accuracy of the LLaMA 2 7B model, which has a cache ratio of 0.2, increased by an average of 7.8% in a 1-shot test and by an average of 5.1% in a 0-shot test compared to H2O through A2SF.

# 2 Related Works

### 2.1 Token pruning for Transformer Encoder

Prior to the exploration of token pruning in Transformer Decoder models, significant research was conducted on token pruning in Transformer Encoder models such as BERT [\[10\]](#page-9-7). These encoder models process all tokens simultaneously, and their primary use was to decrease computational load rather than to manage memory.

Longformer [\[11\]](#page-9-8) introduced predetermined mask for the attention operation. This included Sliding Window Attention, which only considers tokens within a certain distance from the current token, and Dilated Sliding Window Attention. These methods provided a straightforward approach to token pruning. However, since the same token pattern is applied irrespective of the input, there is a potential for accuracy loss due to the inability to consider the context.

In contrast, SpAttn proposed a token pruning technique that utilizes Accumulative Attention Score (A2S). A2S is a method for determining token importance that accumulates Attention Score, which is generated during the attention operation, in units of Key direction tokens. SpAttn further accumulates this A2S at the layer level, establishes the token importance after each layer's operation, and suggests a method to exclude tokens with low A2S from the input of the subsequent layer. This approach is based on that in human language, many tokens, such as prepositions, articles, and adverbs, carry less semantic weight.

### 2.2 Token pruning for Transformer Decoder

The popularity of generative LLM models such as GPT [\[12\]](#page-9-9) based on Transformer Decoder has been on the rise recently. Consequently, token pruning research for Decoder models has emerged as a significant issue. Unlike Encoder models, Decoder models store information about previous tokens in the form of a KV Cache, thereby token pruning reduces not only the computational load but also memory usage. Therefore, the application of the token pruning technique has become an essential requirement for rapidly utilizing LLM in various environments.

Sparse Transformers [\[13\]](#page-9-10) applied a fixed pattern method, similar to the previous Encoder token pruning technique, to LLM. The proposed pattern is a suitable mix of Strided Attention and Fixed Attention. However, this technique also employs a pattern that is independent of the input, so it cannot consider the context at all, leading to a decrease in accuracy.

StreamingLLM [\[14\]](#page-9-11) identified Attention Sink, a phenomenon where the Attention Score is concentrated on the first token during the attention operation of LLM. It also demonstrated that maintaining this Sink Token plays a crucial role in preserving the performance of LLM. Utilizing this, it was confirmed that accuracy could be significantly improved merely by retaining additional Sink Tokens in the above fixed patterns.

In H2O, the intuition of unnecessary tokens was analyzed more experimentally in Decoder models. Similar to Encoder models, most of the Attention Scores are produced by a few tokens, referred to as *Heavy-Hitter*. Also, when these H2 tokens were removed, the accuracy dropped significantly, proving that they play a vital role in the token processing process. And, H2O proposed to apply Accumulative Attention Score-based Token Pruning to prune the KV Cache. However, since there are differences in the token processing process between Encoder models and Decoder models, the method of accumulating A2S across layers was changed to accumulate along the Generation Step. That is, the Attention Score calculated when each token is generated is cumulatively added to the

A2S of each token, and some tokens with small A2S values are removed from the KV Cache. It also proposed head-wise Token Pruning, which maintains A2S in each head and therefore removes different Tokens, improving the existing technique of applying the same Token Pruning Mask in one layer. Through this, it was confirmed that A2S operates quite well in Decoder models.

Subsequent research uses the token selection technique of H2O, but points out that H2O merely deletes unimportant tokens. In response to this, No Token Left Behind [7] proposed to convert unimportant tokens to low bits through Quantization, and Get More with LESS [8] proposed to apply Low-rank Decomposition. Both papers propose methods about processing unimportant tokens after token selection, so they can be used together with A2SF in this paper to improve overall performance.

Keyformer [9] pointed out that after Token Pruning, the denominator of Softmax changes because tokens are removed, and therefore the distribution changes. To solve this problem, it proposed to use Gumbel-Softmax [15], which can flatten the distribution. This technique is also compatible with A2SF because they use A2S after Gumbel-Softmax.

Other research [16, 17] introduces additional learning in the process of selecting important tokens. However, LLM requires a lot of computation and memory during the learning process due to its size, which can be challenging to apply in environments with insufficient computing power and memory.

#### 3 Method

#### 3.1 Accumulative Attention Score

The A2S used in the Encoder model is defined as follows:

$$A_k^l = \sum_{i=1}^l \sum_{h=1}^H \sum_{q=1}^N S_{q,k}^{i,h}$$
 (1)

Here,  $A_k^l$  is the A2S for the kth token in the lth layer, N is the length of the entire sequence, H is the total number of heads, and  $S_{q,k}^{i,h}$  is the Attention Score calculated for the qth query and the kth key in the kth head of the kth layer. This A2S is calculated from the front layers as the operation progresses, allowing each token's importance to be set in that layer. Subsequently, tokens with low importance can be removed from the input of the next layer, reducing the amount of computation.

In H2O, the following changes were made to apply A2S to the Decoder model:

$$A_{n,k}^{l,h} = \sum_{q=1}^{n} S_{q,k}^{l,h} \tag{2}$$

Here, k < n, and by changing the overall accumulation direction to the Generation step instead of the layer, separate scores were maintained at the layer and head level. The above l and h mean the hth head of the lth layer. Then,  $A_{n,k}^{l,h}$  is the A2S of the kth token in the Generation Step where the nth token is generated, and  $S_{q,k}^{l,h}$  is the Attention Score of the kth token in the Step where the qth token is generated. Subsequently, the KV Cache of tokens with small A2S values is removed in the next Generation Step.

However, due to the introduction of the Casual Mask of Masked Self-Attention, the kth token does not perform Attention operations with the previous tokens:

$$S_{q,k}^{l,h} = 0, \forall q < k \tag{3}$$

Therefore, the actual A2S applied to the Decoder model is as follows:

$$A_{n,k}^{l,h} = \sum_{q=k}^{n} S_{q,k}^{l,h} \tag{4}$$

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 2: Toy examples illustrate the Accumulative Attention Score and the Accumulative Attention Score with Forgetting Factor.

Through this, n − k Attention Scores are accumulated in the A2S of the kth token, and n − k − 10 Attention Scores are accumulated in the A2S of the k + 10th token. That is, there is a difference of 10 Attention Scores between these two tokens. The Sof tmax operation does not output negative values due to its characteristics, so if more accumulations occur, the value is likely to be large, which means that the A2S of the token that appeared first is high and is less likely to be removed during Pruning.

### 3.2 Motivation

Figures [1\(](#page-1-0)a) and (b) display the results of applying A2S to BERT and LLaMA, respectively. The top 5 tokens in terms of importance are highlighted in red. In the BERT model (Figure [1\(](#page-1-0)a)), since the Casual Mask was not applied, the Attention Score is evenly distributed among all tokens. This is because the number of accumulations is the same for all tokens, allowing them to be compared fairly based solely on the difference in Attention Score. Conversely, in the case of LLaMA 7B (Figure [1\(](#page-1-0)b)), the importance of the initial tokens is abnormally high due to the effect of the Casual Mask. Also, the importance aligns with the order of generation, which can be considered absolute due to the number of accumulations according to the order of generation rather than the difference in Attention Score. Furthermore, even if the sequence lengthens, there is no change in the previously accumulated Score, making the progress of the sentence and the selected token irrelevant, and the token that was initially generated is always selected.

This differs from human language processing. Human language is written in one paragraph or one sentence, and after moving on, only the important keywords from the previous paragraph are remembered. This implies that humans naturally forget unnecessary parts that appeared in the past during the process of language processing. We derived from this point that the past Attention Score should also be forgotten over time. Also, we adopted a method of setting a Forgetting Factor and multiplying it exponentially based on the well-known human Forgetting study, Ebbinghaus's Forgetting Curve [\[18\]](#page-10-3). This reflects the fact that the simplified form of the Curve is exponential.

#### 3.3 Accumulative Attention Score with Forgetting Factor

We propose to modify the Accumulative Attention Score by introducing a Forgetting Factor α as follows:

$$A_{n,k}^h = \sum_{q=1}^n \alpha^{n-q} \times S_{q,k}^h \tag{5}$$

$$A_{n,k}^{h} = S_{n,k}^{h} + \alpha \cdot S_{n-1,k}^{h} + \alpha^{2} \cdot S_{n-2,k}^{h} + \dots + \alpha^{N-k} \cdot S_{k,k}^{h}$$
 (6)

Here, α is a float number satisfying 0 < α < 1. Because the Forgetting Factor is repeatedly multiplied by the Attention Score calculated in the past Generation Step, its value converges to 0. Therefore, even if many values are accumulated, the value becomes smaller, resolving the imbalance with tokens with fewer accumulations. This can be seen in Figure [2.](#page-4-0) In the case of the second token, it outputs a low Attention Score after its first self-attention and becomes an unnecessary token, but in A2S, it can be seen that it is not deleted due to the influence of the past. On the other hand, in A2SF, the

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 3: Comparison of accuracy between Full cache, Local Attention, and H2O

scores in the upper left, which are the Attention Scores created in the past from the old tokens, can be seen to gradually converge to 0 due to the influence of the Forgetting Factor. Through this, it can be confirmed that the token that is most unnecessary at the current point is removed, not the token that was removed with fewer accumulations. The results in actual data can also be confirmed through Figure [1\(](#page-1-0)c). Unlike Figure [1\(](#page-1-0)b) where A2S was applied, it can be seen that the distribution of scores between tokens is even. Through this, tokens can be compared more fairly in terms of importance, and more accurate results can be output by selecting actually important tokens.

Also, A2SF has the potential for tuning considered for the dataset by setting the value of α. If a value close to 1 is assigned to α, convergence occurs slowly, so the Attention Score that occurred in the past can still affect the present, but if α is close to 0, it converges quickly, so the Attention Score calculated recently is used to determine the importance of the token, which means comparing tokens using only recent trends. In other words, depending on the characteristics of the dataset, the degree of influence of the past on the present can be adjusted, and it means that the optimal accuracy for the situation can be achieved by setting an appropriate α.

# 4 Experiments

#### 4.1 Experimental setup

For the accuracy measurement experiment of A2SF, we used the LLaMA-2-7B [\[19\]](#page-10-4), LLaMA-7B [\[20\]](#page-10-5), OPT-6.7B, OPT-2.7B [\[21\]](#page-10-6) models, and evaluated the Commonsense-reasoning performance with the dataset of OpenbookQA [\[22\]](#page-10-7), Winogrande [\[23\]](#page-10-8), PiQA [\[24\]](#page-10-9), COPA [\[25\]](#page-10-10), MathQA [\[26\]](#page-10-11), ARC-easy, ARC-challenge [\[27\]](#page-10-12). We used lm-eval-harness [\[28\]](#page-10-13)(v0.4.0) for evaluation. The experiment was conducted with an FP16 model in an RTX 3090 environment. In H2O, Local Attention and token selection through A2S are used together, and each ratio is set to half of the cache ratio. Since A2SF includes considering recent Attention Score trends in the algorithm, we did not allocate a separate local cache ratio and allocated all to selective cache ratio.

### 4.2 Model accuracy

We measured the accuracy of a total of 4 models in 1-shot and 0-shot tasks for cache ratios in the range of [0.1, 0.8] across 7 datasets. Among them, the results for the 1-shot task are summarized in Figure [3.](#page-5-0) Through this, we were able to draw the following conclusions: (1) In most cases, A2SF outperforms H2O in terms of accuracy. (2) At a cache ratio of 0.4 or higher, it can approach the accuracy of Ideal under many conditions. (3) There are cases where the limit of accuracy that can be achieved with a small number of tokens is low, depending on the dataset. The full experimental results of LLaMA 2 7B are in Table [1.](#page-6-0) We were able to confirm an accuracy improvement of 7.8% in the 1-shot environment and 5.1% in the 0-shot environment compared to H2O.

<span id="page-6-0"></span>

| Shots  | Method | Dataset |       |       |       |        |       |       |         |
|--------|--------|---------|-------|-------|-------|--------|-------|-------|---------|
|        |        | OBQA    | WG    | PiQA  | COPA  | MathQA | ARC-E | ARC-D | Average |
| 1-shot | Full   | 0.362   | 0.691 | 0.774 | 0.850 | 0.303  | 0.786 | 0.467 | 0.605   |
|        | Local  | 0.122   | 0.487 | 0.574 | 0.630 | 0.207  | 0.370 | 0.225 | 0.374   |
|        | H2O    | 0.224   | 0.511 | 0.726 | 0.660 | 0.202  | 0.575 | 0.310 | 0.458   |
|        | A2SF   | 0.268   | 0.526 | 0.756 | 0.820 | 0.251  | 0.724 | 0.410 | 0.536   |
| 0-shot | Full   | 0.324   | 0.688 | 0.781 | 0.860 | 0.282  | 0.763 | 0.423 | 0.589   |
|        | Local  | 0.130   | 0.487 | 0.540 | 0.640 | 0.194  | 0.330 | 0.215 | 0.362   |
|        | H2O    | 0.152   | 0.512 | 0.676 | 0.570 | 0.210  | 0.419 | 0.253 | 0.399   |
|        | A2SF   | 0.176   | 0.526 | 0.711 | 0.620 | 0.221  | 0.586 | 0.311 | 0.450   |

Table 1: Comparison of the performance of existing techniques and A2SF for different number of shots in LLaMA 2 7B, with a Cache Ratio of 0.2

### 4.3 Token selection

Figure [4](#page-6-1) depicts the Attention Score after applying each technique with a cache ratio of 0.2. The *Ideal* in Figure [4\(](#page-6-1)a) calculates the Attention Score of all tokens at each sequence generation step (each row), selects tokens with large values, and considers all tokens in the subsequent step without removing tokens with small values. This represents the Mask with the largest Attention Score given the Cache Size. However, since tokens are not removed, there is no compression effect, making it an ideal method. Therefore, the closer the result of applying the Token Pruning technique is to this Ideal Mask, the closer it is to achieving ideal accuracy.

Local Attention (Figure [4\(](#page-6-1)b)) exhibits a lower mask similarity because it only considers tokens of a certain size based on the most recent tokens. Also, due to the absence of an Attention Sink, a significant change in the distribution of Softmax can be observed. H2O (Figure [4\(](#page-6-1)c)) creates a λ-shaped Mask due to the issue of high scores of initial tokens, which differs from the Ideal Mask. In contrast, when A2SF was applied, a Mask is similar to the Ideal Mask compared to existing techniques. Moreover, the Attention Sink Token, which can significantly impact accuracy, is selected without any issues. This is because even when the Forgetting Factor is applied, the Sink Token is continuously calculated with a large value, allowing it to maintain a high value compared to other tokens.

<span id="page-6-1"></span>![](_page_6_Figure_5.jpeg)

Figure 4: Attention Score after applying each technique. The intensity of the color is directly proportional to the value it represents; a darker color signifies a higher value.

In Table [2,](#page-7-0) the average cosine similarity between the Attention Score of techniques and the Ideal is presented. Local Attention, where the Attention Score distribution is disrupted, shows a very low similarity. Although the Sink Token is included, H2O exhibits a relatively low similarity because the area in the middle of the λ disappears. On the other hand, the Mask created through A2SF has an average similarity close to 1. This result explains why the model's accuracy increases when A2SF is applied.

<span id="page-7-0"></span>

| Method   | WG    | PiQA  | OBQA  | ARC-E | Average |  |
|----------|-------|-------|-------|-------|---------|--|
| Local    | 0.320 | 0.315 | 0.317 | 0.318 | 0.318   |  |
| H2O      | 0.960 | 0.970 | 0.968 | 0.970 | 0.967   |  |
| A2SF 0.1 | 0.990 | 0.992 | 0.990 | 0.991 | 0.991   |  |
| A2SF 0.5 | 0.988 | 0.991 | 0.987 | 0.988 | 0.989   |  |

Table 2: Average cosine similarity between the Attention Score generated by each method and the Attention Score applied with Ideal Pruning. Average cosine similarity between the Attention Score generated by each method and the Attention Score applied with Ideal Pruning.

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 5: Factor graph

#### 4.4 Discussion

#### 4.4.1 Determining the optimal Forgetting Factor value

Figure [5](#page-7-1) illustrates the accuracy achieved when applying a Forgetting Factor within the range of [0.0, 0.9], with a fixed cache ratio of 0.3. Generally, the peak accuracy is observed within the Forgetting Factor range of [0.1, 0.3], and the accuracy tends to decline as the Factor increases. A relatively low Forgetting Factor implies that the model considers a shorter history. This observation suggests that when applying the A2S to the Transformer Decoder model, an extensive historical context can have a detrimental effect.

However, in the MathQA dataset, there were instances where a relatively high Forgetting Factor resulted in high accuracy, and in some cases, even the highest accuracy. This discrepancy can be attributed to the differences between the datasets. Here are some examples of sequences included in each dataset:

### • OpenbookQA

– If you wanted to make a necklace, how long would you have to wait for the materials to appear inside the Earth? Millions of years

#### • PiQA

– Question: What ingredient is left out of fluffy slime that is normally in regular slime? Answer: Borax

## • MathQA

– Question: In a 160 meters race, a beats b by 56 m or 7 seconds. What is a's time over the course? Answer: 22 seconds

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 6: Attention Score, A2S and A2SF of the important tokens according to the Generation Step. The starting point of the graph differs because each token is generated at different steps.

Compared to other datasets, a distinctive feature of MathQA is its clear problem structure due to the use of numbers. In other words, the distinction between necessary and unnecessary tokens is evident, suggesting that it could be beneficial to consider a longer history and prevent the removal of important tokens.

### 4.4.2 Influence of history

In Figure [5,](#page-7-1) we observed instances where the highest accuracy was attained when the Forgetting Factor was set to 0.0. A Forgetting Factor of 0.0 implies that the model does not consider history at all, and it determines the importance of a token solely based on the Attention Score from the immediate preceding sequence. This approach demonstrated higher accuracy across all cases compared to H2O, which takes into account all history. This suggests that for the Decoder model, it is more beneficial to disregard history entirely rather than considering an excessive amount of it. However, we generally observed higher accuracy when the model considered recent history through a low Forgetting Factor in the range of [0.1, 0.3], indicating that history can serve as a supplementary factor.

Figure [6](#page-8-0) illustrates the Attention Score generated by the primary tokens as the Generation Step progresses, along with the A2S and A2SF calculated from it. This figure reveals that the remaining tokens, excluding the Attention Sink, do not consistently have a large Score, even if they are major tokens. In this process, history plays a role in preventing a token, which had a large Score, from being eliminated during a certain Step by compensating the value when it outputs a small Score, or awarding extra points when it continuously outputs a large Score. Since A2S maintains the accumulated value from the past, a token that was initially created carries a high level of importance, even if the Score generated over time is small. If Token Pruning is performed at the point indicated by the red line in Figure [6,](#page-8-0) *Answer* is removed despite its high Attention Score due to its small accumulated value. In contrast, A2SF accounts for the reduced Score of past tokens, so the importance of the *Answer* remains high.

# 5 Conclusion

We propose a novel method, A2SF (Accumulative Attention Score with Forgetting Factor), to rectify imbalances in the Accumulative Attention Score utilized for token importance evaluation in Transformer Decoder-based models. A2SF addresses the skewed importance scoring induced by the causal masking in Masked Self-Attention, which disproportionately amplifies the significance of early sequence tokens. A2SF incorporates a Forgetting Factor to mitigate this historical bias in Attention Scores, thereby ensuring a more equitable distribution of token importance across the sequence length. Comprehensive experiments on a variety of models and datasets demonstrate that A2SF surpasses existing methods, enhancing accuracy without necessitating additional model retraining. Furthermore, given that A2SF is a token selection technique, it holds potential to enhance the performance of numerous ongoing KV Cache processing algorithms when integrated with A2SF.

# 6 Limitation

We applied a uniform value for the Forget Factor across all heads, layers, and tokens. However, the characteristics of these elements can vary significantly. For instance, tokens can represent diverse features such as nouns, verbs, and so on. Therefore, applying the same rate of forgetting to all elements may not yield optimal results. For example, when summarizing content, it could be beneficial to retain as much information as possible about unique nouns that form the core of a paragraph. To address this, we could devise methods to determine an appropriate Forget Factor based on the form of the Attention Score generated during previous generation steps. Alternatively, we could set a distinct Forget Factor for each layer, head, and token by understanding their characteristics in advance.

# References

- <span id="page-9-0"></span>[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-9-1"></span>[2] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. Flexgen: High-throughput generative inference of large language models with a single gpu. In *International Conference on Machine Learning*, pages 31094–31116. PMLR, 2023.
- [3] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th Symposium on Operating Systems Principles*, pages 611–626, 2023.
- <span id="page-9-2"></span>[4] Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. Accelerate: Training and inference at scale made simple, efficient and adaptable. <https://github.com/huggingface/accelerate>, 2022.
- <span id="page-9-3"></span>[5] Hanrui Wang, Zhekai Zhang, and Song Han. Spatten: Efficient sparse attention architecture with cascade token and head pruning. In *2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, pages 97–110. IEEE, 2021.
- <span id="page-9-4"></span>[6] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. *Advances in Neural Information Processing Systems*, 36, 2024.
- <span id="page-9-5"></span>[7] Roni Paiss, Hila Chefer, and Lior Wolf. No token left behind: Explainability-aided image classification and generation. In *European Conference on Computer Vision*, pages 334–350. Springer, 2022.
- <span id="page-9-12"></span>[8] Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. Get more with less: Synthesizing recurrence with kv cache compression for efficient llm inference. *arXiv preprint arXiv:2402.09398*, 2024.
- <span id="page-9-6"></span>[9] Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant J Nair, Ilya Soloveychik, and Purushotham Kamath. Keyformer: Kv cache reduction through key tokens selection for efficient generative inference. *arXiv preprint arXiv:2403.09054*, 2024.
- <span id="page-9-7"></span>[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.
- <span id="page-9-8"></span>[11] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020.
- <span id="page-9-9"></span>[12] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.
- <span id="page-9-10"></span>[13] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*, 2019.
- <span id="page-9-11"></span>[14] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. *arXiv preprint arXiv:2309.17453*, 2023.

- <span id="page-10-0"></span>[15] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. *arXiv preprint arXiv:1611.01144*, 2016.
- <span id="page-10-1"></span>[16] Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, and Thomas Hofmann. Dynamic context pruning for efficient and interpretable autoregressive transformers. *Advances in Neural Information Processing Systems*, 36, 2024.
- <span id="page-10-2"></span>[17] Piotr Nawrot, Adrian Łancucki, Marcin Chochowski, David Tarjan, and Edoardo M Ponti. ´ Dynamic memory compression: Retrofitting llms for accelerated inference. *arXiv preprint arXiv:2403.09636*, 2024.
- <span id="page-10-3"></span>[18] Piotr Wo´zniak, Edward Gorzelanczyk, and Janusz Murakowski. Two components of long-term ´ memory. *Acta neurobiologiae experimentalis*, 55(4):301–305, 1995.
- <span id="page-10-4"></span>[19] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-10-5"></span>[20] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.
- <span id="page-10-6"></span>[21] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*, 2022.
- <span id="page-10-7"></span>[22] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *EMNLP*, 2018.
- <span id="page-10-8"></span>[23] Winogrande: An adversarial winograd schema challenge at scale. 2019.
- <span id="page-10-9"></span>[24] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In *Thirty-Fourth AAAI Conference on Artificial Intelligence*, 2020.
- <span id="page-10-10"></span>[25] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In *2011 AAAI Spring Symposium Series*, 2011.
- <span id="page-10-11"></span>[26] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2357–2367, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
- <span id="page-10-12"></span>[27] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv:1803.05457v1*, 2018.
- <span id="page-10-13"></span>[28] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023.