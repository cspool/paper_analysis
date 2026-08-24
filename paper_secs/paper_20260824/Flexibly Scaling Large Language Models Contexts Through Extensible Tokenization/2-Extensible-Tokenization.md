# 2 Extensible Tokenization

#### 2.1 Framework

The workflow of Extensible Tokenization is shown as Figure [2.](#page-2-0) For each long-sequence input X, we perform the following three steps which enables the long context to be utilized by the LLM. Firstly, the input X is chunked into sub-sequences: {X1, ...X<sup>N</sup> }. The sequence length of each chunk L<sup>i</sup> is set to be the maximum window size of the extensible tokenizer, e.g., L<sup>i</sup> = 4096 with LLaMA-2, which will best preserve the coherence of the chunking result. Secondly, the sub-sequence of each chunk is transformed by the extensible tokenizer into the output embeddings. The output embeddings are down-scaled by the scaling factor k (e.g., k = 16 or 32), where L/k extensible embeddings (denoted as ET) are generated as the condensed representation of the raw input. Finally, the new tokens are predicted conditioned on the extensible embeddings from the preceding chunks and the raw token embeddings within the recent context.

> **[图片提取文字 (无描述)]:**
> Inference pass for super tokens Decoding pass for x<sub>4-6</sub> and x<sub>7-9</sub> Decoding pass for x<sub>10-12</sub> and x<sub>13-15</sub>  $\{et_1: x_{1-3}, et_2: x_{4-6}, et_3: x_{7-9}, et_4: x_{10-12}\}$ et<sub>1</sub> et, Extensible Token Embeddings {et<sub>1</sub> ··· et<sub>4</sub>} } et<sub>2</sub> et2 et<sub>3</sub> et<sub>3</sub> et<sub>4</sub> eta X4 X10 Extensible Tokenizer X11 X5 x6 X12 X7 X13 X8 X14 Raw Token Embeddings {x1 ··· x12} Xq X<sub>15</sub>
![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Figure 3: Two-Stream AR. In the first pass, the raw token embeddings are transformed into extensible embeddings (with the scaling factor k = 3). In the second pass (given a window size of 10), the auto-regression is accomplished in two steps, with the x1−<sup>3</sup> and x4−<sup>6</sup> predicted in the first step, and x7−<sup>9</sup> and x10−<sup>12</sup> predicted in the second step.

#### 2.2 Extensible Embedding

As introduced, the raw token embedding, which is merely corresponding to one individual token, is information sparse. In contrast, the extensible embedding is presented to serve as a highly compact but equally informative representation of the context. For this purpose, we employ another language model as the extensible tokenizer (denoted as LMet), which transforms the raw input of each subsequence X<sup>i</sup> : {xi,1, ...xi,L} into the sequence of output embeddings O<sup>i</sup> :

$$O_i: \{o_{i,1}, ..., o_{i,L}\} \leftarrow LM_{et}(x_{i,1}, ..., x_{i,L}; \theta_{et}).$$
 (1)

On top of an expressive language model, the rich contextual information within Xi,:<sup>l</sup> can be encoded by the corresponding output embedding oi,l. The output embeddings are further down-scaled by the scaling factor k, where m (m = L/k) extensible embeddings (eti,<sup>∗</sup>) are generated for X<sup>i</sup> :

$$\{et_{i,1}, \dots, et_{i,m}\} \leftarrow \text{DownScale}(\{o_{i,1}, \dots o_{i,L}\}). \tag{2}$$

There can be many alternative ways to realize the functionality of down-scaling, e.g., with any pooling functions along the sequence dimension. In our work, we simply down-scale the output embeddings by through strided sampling, where the last embedding in every k steps is chosen, i.e., eti,j ← oi,k×<sup>j</sup> . Despite simplicity, such a realization is empirically effective and leads to a high flexibility of usage.

#### 2.3 Two-Stream AR

Extensible Tokenization can be learned by auto-regression (AR), where the loss is minimized for the prediction of next tokens conditioned on the extensible embeddings from the preceding context. Although the auto-regression tasks can be simply performed by having the long context transformed into extensible embeddings and predicting the last few tokens within a training instance (e.g., predicting the answer to a question based on the extensible embeddings of a long document), it will severely restrict the training effect because the long context accounts for the majority of computation cost whereas no prediction loss can be derived from it.

In our work, we propose the two-stream AR to optimize the sample efficiency of training (Figure [3\)](#page-3-0). In the first pass of inference, the extensible embeddings are generated for the entire context. For example, with a chunk size of 3 and an scaling factor of 3, the input data X = {x1, ...x15} is transformed into the extensible embeddings {et1,1, et2,1, et3,1, et4,1} (the last chunk is exempted). In the second past, each single token within the long context is streamingly predicted by chunks. Particularly, the prediction is made conditioned on the extensible embeddings from the previous chunks and the preceding raw token embeddings within the same chunk. Formally,

$$\min_{\theta_{et}} \sum_{X} \sum_{i>1} \log P(x_{i,j}|et_{1,1}, ...et_{i-1,k}, x_{i,1}, ...x_{i,j-1}|\theta, \theta_{et}).$$
 (3)

For example, x<sup>6</sup> is predicted based on st<sup>1</sup> (representing x1−3) and x4. Note that the chunk size of training is made much smaller than the LLM's window size (e.g., 512), where the prediction of new tokens can mostly rely on the contextual information offered by the extensible embeddings. Thanks to the above processing, the prediction loss can be comprehensively derived from the each training

instance, enabling the extensible tokenizer to be effectively learned from general long-context data, such as RedPajama [\[10\]](#page-9-11) and LongAlpaca [\[6\]](#page-9-0). We also randomly sample the extension ratio k from a candidate scope (e.g., [2, 4, 8, 16, 32]) for each training instance, which helps the model to generalize for the extension of diverse context lengths.

When the extensible tokenizer is learned, the downstream LLM's parameters (θ) are always fixed. Consequently, the extensible tokenizer can work as a compatible drop-in component to the downstream LLM, bringing in new information from the extended context without affecting the LLM's performance with the raw token embeddings. Besides, we also empirically find that the extensible tokenizer can maintain a high compatibility with many full-parameter fine-tuned derivatives of the downstream LLM. Such a property makes it a general module to extend the context length for a family of closely related LLMs.

