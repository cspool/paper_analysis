# <span id="page-7-0"></span>**5 Experiments**

We briefly describe the data and model configurations used to identify the *forget threshold T*forget and *maximum recall context length T*recall. Due to limited space, more comprehensive experimental details are reported in Appendix [F.](#page-15-0)

**Data** We start from RedPajama-V2 [\(Computer,](#page-9-5) [2023\)](#page-9-5), an open dataset with 30T tokens from the Internet, and perform deduplication to ensure data quality and discard documents that are too short.

**Models** We experiment with six model sizes to find the relationship between state capacity and size. For each of them, we perform an extensive search with training lengths up to 256K tokens. To save cost, we continue pre-training from three official checkpoints of Mamba-2 (130M, 370M, and 780M). They were pre-trained with 8K sequences. The other model configurations (36M, 47M, and 85M) are trained from scratch.

### <span id="page-7-1"></span>**5.1 The Existence of Forget Threshold**

In Figure [10,](#page-7-2) we plot the language modeling perplexity as a function of token position for Mamba-2 130M and 370M with different training lengths. We can see that for each model size, there is a training length threshold, beyond which the model has much better length extrapolation, which supports our arguments discussed in Section [4.2.](#page-6-3)

#### **5.2 Forget Threshold as a Function of the State Size**

Figure [11](#page-7-2) shows the minimum training length needed for Mamba-2 to learn forgetting. The rightmost data point in the plot corresponds to Mamba-2 370M. We have confirmed that the 780M model (with a state size of 19.3M) also has poor length generalization at training lengths below 128K, but do not have enough resources to train the model beyond this length. The results establish a linear relationship *T*forget = 5.172 · *N<sup>S</sup>* − 4.469 between the length *T*train = *T*forget at which the model can learn robust forgetting and the state size *NS*. The *R* <sup>2</sup> value is over 0.999. This indicates that **to train a Mamba-2 with robust length generalization, one should use training lengths that grow linearly with the state size.**

#### **5.3 Maximum Recall Context Length as a Function of the State Size**

The second plot of Figure [9](#page-6-0) shows the recall threshold of Mamba-2 in passkey. The maximum contexts length in which Mamba-2 can accurately retrieve 5-digit passkeys is exponential concerning the state size, the function is *T*recall = 4.756 · (1.365*N<sup>S</sup>* − 1) − 0.742, with an *R* 2 value over 0.999. This is because the amount of information in the context does not increase with its length. In other words, we are storing a constant amount of information while the

number of combinations of the state grows exponentially with the number of elements. The result is very promising because, to the best of our knowledge, no previous models with less than 1B model parameters have near-perfect accuracy at this length in this task.

## **6 Related Works**

**RNN-Based Language Models** This paper focuses on Mamba-2, a recurrent architecture that can be viewed as a variant of gated linear attention [\(Yang et al.,](#page-11-1) [2024a\)](#page-11-1). Many recently proposed RNNs can also be viewed as GLA variants. These include the RWKV series [\(Peng](#page-10-6) [et al.,](#page-10-6) [2023;](#page-10-6) [2024a\)](#page-10-3), the Mamba series [\(Gu & Dao,](#page-10-2) [2023;](#page-10-2) [Dao & Gu,](#page-9-3) [2024\)](#page-9-3), GLA [\(Yang et al.,](#page-11-1) [2024a\)](#page-11-1), and many more [\(Zhang et al.,](#page-11-5) [2024b;](#page-11-5) [Yang et al.,](#page-11-6) [2024b;](#page-11-6) [De et al.,](#page-9-6) [2024;](#page-9-6) [Arora et al.,](#page-9-7) [2024b;](#page-9-7) [Orvieto et al.,](#page-10-7) [2023;](#page-10-7) [Sun et al.,](#page-10-5) [2023\)](#page-10-5). Our methods may apply to these architectures as well. Some recent/concurrent RNNs such as Gated DeltaNet [\(Yang et al.,](#page-11-7) [2025\)](#page-11-7), RWKV-7 [\(Peng et al.,](#page-10-8) [2025\)](#page-10-8), xLSTM [\(Beck et al.,](#page-9-8) [2024\)](#page-9-8), and Titans [\(Behrouz et al.,](#page-9-9) [2024\)](#page-9-9) have gone beyond a gating-based memory decay mechanism and are out of the scope of this paper.

**Length Generalization** Most SOTA language models in the last few years have been based on the transformer [\(Vaswani et al.,](#page-11-0) [2017\)](#page-11-0) architecture. These models, when using certain variants of position encoding, can process arbitrarily long sequences. However, they exhibit severe performance drops on tokens beyond the training length [\(Zhao et al.,](#page-11-8) [2024\)](#page-11-8). To alleviate this shortcoming, many works have focused on modifying positional encoding [\(Peng et al.,](#page-10-9) [2024b;](#page-10-9) [Zhu et al.,](#page-11-9) [2024;](#page-11-9) [Ding et al.,](#page-9-10) [2024;](#page-9-10) [Jin et al.,](#page-10-10) [2024\)](#page-10-10), some achieving training-free length generalization to certain extents [\(Zhang et al.,](#page-11-10) [2025\)](#page-11-10).

**Length Generalization of Mamba** Some prior works investigated the performance of Mamba as a function of context length [\(Park et al.,](#page-10-11) [2024;](#page-10-11) [Wen et al.,](#page-11-11) [2025\)](#page-11-11). [Jelassi et al.](#page-10-12) [\(2024\)](#page-10-12) empirically showed a sharp performance drop beyond the training length for Mamba on a copying task and also showed that Mamba struggles to copy from context unless its state size grows linearly with the context length. [Arora et al.](#page-9-11) [\(2024a\)](#page-9-11) discussed the associative recall abilities of transformer and some RNNs. [Wang et al.](#page-11-12) [\(2025\)](#page-11-12) is most related to our work. They discussed the issue of over-smoothing introduced by the memory decay term. In contrast, our paper explores a setting where recency may be preferred, but interference from earlier tokens damages the recall accuracy of recent tokens.

Some concurrent works have explored extending Mamba's context length by controlling the discretization term (∆*<sup>t</sup>* in Eq. [2\)](#page-2-2) [\(Ben-Kish et al.,](#page-9-4) [2024\)](#page-9-4), such as dividing it by a constant to make it smaller [\(Zhang,](#page-11-4) [2023\)](#page-11-4). This makes the memory decay factor (*αt* in Eq. [4\)](#page-2-3) closer to 1, which makes the state retain more contextual information. However, it also unnecessarily diminishes the inserted information on all tokens. Similar to the above works, this study explores the cause of Mamba-2's inability to generalize beyond its training context length and provides valuable insights into training Mamba-2 models that generalize better.

## **7 Conclusion**

This paper demonstrates that while the Mamba architecture includes a memory decay mechanism, it fails to effectively learn forgetting in practice. As a result, when the context exceeds the training length, the model produces incoherent outputs. This issue arises from training with contexts that are too short relative to the state size. Empirical results show that robust forgetting is only learned when the training context length surpasses a certain threshold, which increases linearly with the state size. Notably, the model is still capable of recalling some contextual information beyond this threshold. These findings offer valuable insights into the causes and consequences of the model's inability to forget, highlighting key limitations of the Mamba architecture. Nevertheless, the insights gained from this study provide a promising foundation for improving Mamba's performance in long-context modeling, paving the way for more effective applications in tasks requiring extended context lengths.

## **Acknowledgements**

This work is supported by the high-quality development project of MIIT and a grant from the Guoqiang Institute, Tsinghua University.

This work is supported by the National Natural Science Foundation of China (No. 623B2065)

