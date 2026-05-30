# **C.4 Performance on MoE architectures**

We conducted additional experiments using the Qwen1.5-MoE-14.3B-A2.7B [\(Qwen, 2024a\)](#page-13-15) model on tasks within LongBench. As shown in Table [10,](#page-19-0) when replacing 50% of its layers with streaming attention, LightTransfer exhibits a performance drop of less than 1% on the MoE architecture, outperforming other layer-wise KV cache pruning baselines. This observation is consistent with our findings on non-MoE transformer models, thereby further confirming the robustness and effectiveness of our approach.

### <span id="page-18-1"></span>**C.5 Comparasion with other layer layer replacement strategies**

To further study the effectiveness of our approach, we benchmark it against the following two baselines :

- (i) **Shapley value–based** [\(Zhang et al., 2024c\)](#page-15-6), which replaces layers according to their estimated Shapley contribution;
- (ii) **BERTology** [\(Rogers et al., 2021\)](#page-13-16), which iteratively substitutes the least influential layers.

Table 9: Comparison with Head-wise KV Cache Reduction Methods on LLaMA3-8B-Instruct-Gradient-1048K.

| Dataset       | Razor | HeadKV | Ours  | DuoAttn |
|---------------|-------|--------|-------|---------|
| Training Free | Yes   | Yes    | Yes   | No      |
| qasper        | 19.69 | 29.97  | 26.85 | 27.02   |
| multiqa_en    | 27.62 | 30.93  | 48.26 | 53.69   |
| hotpotqa      | 23.98 | 26.96  | 37.07 | 35.52   |
| 2wiki         | 24.83 | 25.70  | 30.41 | 28.08   |
| multinews     | 25.84 | 27.31  | 26.50 | 27.76   |
| trec          | 55.50 | 58.50  | 66.00 | 69.00   |
| triviaqa      | 63.74 | 78.54  | 87.68 | 87.32   |
| samsum        | 40.10 | 39.70  | 41.04 | 41.13   |
| pcount        | 2.26  | 0.39   | 2.00  | 2.00    |
| lcc           | 32.19 | 34.54  | 41.20 | 39.24   |
| repo-p        | 32.15 | 36.06  | 40.02 | 40.04   |
| Average       | 31.63 | 35.33  | 40.64 | 40.98   |

Table 10: Performance on Qwen1.5-MoE-14.3B-A2.7B.

<span id="page-19-0"></span>

| Dataset         | Standard | MiniCache | SqueezeAttn | LitTrans |
|-----------------|----------|-----------|-------------|----------|
| qasper          | 30.19    | 18.66     | 23.48       | 26.92    |
| multifieldqa_en | 38.47    | 24.14     | 29.58       | 37.23    |
| hotpotqa        | 10.17    | 5.57      | 8.56        | 9.12     |
| 2wikimqa        | 11.92    | 7.64      | 11.35       | 12.78    |
| multi_news      | 24.77    | 20.09     | 20.94       | 24.63    |
| trec            | 68.00    | 66.00     | 64.50       | 65.50    |
| triviaqa        | 86.35    | 70.44     | 84.80       | 85.35    |
| samsum          | 38.09    | 24.87     | 38.22       | 37.74    |
| passage_count   | 1.67     | 2.04      | 3.30        | 3.10     |
| lcc             | 48.33    | 33.38     | 44.90       | 48.68    |
| repobench-p     | 40.89    | 22.09     | 36.62       | 38.33    |
| Average         | 36.26    | 26.81     | 33.30       | 35.43    |

<span id="page-20-0"></span>![](_page_20_Figure_1.jpeg)

Figure 9: Additional examples of layer behavior across tokens.

Table 11 summarises the performance on three tasks from LongBench. Our **LightTransfer** strategy attains the best performance on all datasets, surpassing the next-best baseline by up to +3.7% (HotpotQA) while maintaining the same inference budget.

<span id="page-20-2"></span>Table 11: Comparison of different layer-replacement strategies on three datasets on LLaMa3-8B-Instruct.

| Method              | HotpotQA | MuSiQue | NarrativeQA |
|---------------------|----------|---------|-------------|
| Shapley value-based | 39.97    | 18.18   | 19.63       |
| BERTology           | 42.48    | 18.13   | 20.91       |
| LightTransfer       | 43.70    | 20.90   | 23.20       |

### D More Examples

#### D.1 Examples about Layer Behavior across Tokens

Additional examples of layer behavior across tokens for a given input can be found in Figure 9. The examples are randomly chosen from LongBench benchmarks. The analysis is conducted using LLaMA3-8B-Instruct.

#### **E** Notation

For a positive integer  $N \in \mathbb{N}$ , we define the set  $[N] = \{1, \cdots, N\}$ . For a vector  $x \in \mathbb{R}^d$ , we adopt  $\|\cdot\|_p$  to denote the  $\ell_p$  norm of vectors. For a matrix  $X = [x_1^\top, \cdots, x_{d_1}^\top]^\top \in \mathbb{R}^{d_1 \times d_2}$ , where  $x_i \in \mathbb{R}^{d_2}$  for  $i = 1, \cdots, d_1$ , we define the  $\ell_{p,q}$ -norm of X as  $\|X\|_{p,q} = \|[\|x_1\|_p, \cdots, \|x_{d_1}\|_p]\|_q$ , i.e., we first apply  $\ell_p$  norm in a row-wise manner and then apply  $\ell_q$  norm. The Frobenius norm  $\|\cdot\|_{2,2}$  is also denoted as  $\|\cdot\|_{\mathbf{F}}$ . For a matrix  $X \in \mathbb{R}^{a \times b}$ , its i-th row and i-th column are denoted as  $[X]_{i,i}$  and  $[X]_{:,i}$ , respectively. The element at i-th row and j-th column of X is denoted as  $[X]_{i,j}$ .

#### <span id="page-20-1"></span>F Theoretical Analysis

In this section, we provide the theoretical analysis of the proposed method. We first define the transformer structure we analyze in this paper. In fact, we analyze the LLaMA-type structure (Dubey et al., 2024), i.e., the transformers that adopt the pre-norm and the res-link. The input of the transformer is the embedding of the tokens  $X \in \mathbb{R}^{N \times d}$ , where N is the number of tokens, and d is the dimension of the token embedding. We consider a L-layer transformer, i.e., there are L transformer blocks in the network. Each transformer block consists of a Multi-Head Attention (MHA) and a Feed-Forward (FF) module. The MHA module is a combination of multiple causal self-attention modules. Each causal self-attention module is defined as

$$\mathsf{attn}(X, W_Q, W_K, W_V) = \mathsf{softmax}(XW_QW_K^\top X^\top + M)XW_V,$$

where  $X \in \mathbb{R}^{N \times d}$  is the input,  $W_Q, W_K \in \mathbb{R}^{d \times d_k}$  and  $W_V \in \mathbb{R}^{d \times d}$  are the weights of the self-attention module, and  $M \in \mathbb{R}^{N \times N}$  is the causal mask. The causal mask is defined as

$$[M]_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{otherwise.} \end{cases}$$

The MHA with H heads is defined as

$$\mathsf{mha}\Big(X, \{W_{Q,h}, W_{K,h}, W_{V,h}\}_{h=1}^H\Big) = \sum_{h=1}^H \mathsf{softmax}(XW_{Q,h}W_{K,h}^\top X^\top + M)XW_{V,h},$$

where  $X \in \mathbb{R}^{N \times d}$  is the input,  $W_{Q,h}, W_{K,h} \in \mathbb{R}^{d \times d_k}$  and  $W_{V,h} \in \mathbb{R}^{d \times d}$  are the weights of the h-th head of MHA. Here we just merge the parameter  $W_O$  into  $W_V$  for ease of notation. Our analysis can be directly applied to the parameterization that explicitly includes  $W_O$  as a weight. The FF module applies transformations to X in a row-wise manner, which can be defined as

$$ffn(X, W_{A,1}, W_{A,2}) = \sigma(XW_{A,1})W_{A,2},$$

where  $W_{A,1}, W_{A,2} \in \mathbb{R}^{d \times d}$  are weights of FF module, and  $\sigma(\cdot)$  is an element-wise activation function. For example,  $\sigma$  can be ReLU function. We require that  $\sigma$  is a Lipschitze function.

<span id="page-21-0"></span>**Assumption F.1.** The activation function  $\sigma(\cdot)$  is  $L_{\text{lip}}$ -Lipschitze, i.e.,  $|\sigma(x) - \sigma(y)| \leq L_{\text{lip}}|x - y|$  for any  $x, y \in \mathbb{R}$ .

We note that this assumption is satisfied by all the popular activation functions, including ReLU, sigmoid, ELU, and GELU. The input of the transformer is denoted as the output of the 0-th layer, i.e.,  $X^{(0)} = X$ . Then the *i*-th block processes in the input  $X^{(i-1)}$  as

$$Y^{(i)} = X^{(i-1)} + \mathsf{mha}\Big(\mathsf{LN}(X^{(i-1)}), \{W_{Q,h}^{(i)}, W_{K,h}^{(i)}, W_{V,h}^{(i)}\}_{h=1}^{H}\Big) \tag{2}$$

$$X^{(i)} = Y^{(i)} + \text{ffn}(\mathsf{LN}(Y^{(i)}), W_{A.1}^{(i)}, W_{A.2}^{(i)}), \tag{3}$$

where the superscript (i) denotes the parameters and hidden states at layer i, and LN is the row-wise normalization of the input. To simplify the mathematical calculation, we defined LN as

$$\mathsf{LN}(x) = \begin{cases} x & \text{if } ||x||_2 \le 1\\ x/||x||_2 & \text{otherwise} \end{cases}$$

Our analysis can be directly applied to the Layer Norm function of PyTorch. For ease of notation, we will abbreviate  $\mathsf{mha}(\cdot, \{W_{Q,h}^{(i)}, W_{K,h}^{(i)}, W_{V,h}^{(i)}\}_{h=1}^{H})$  and  $\mathsf{ffn}(\cdot, W_{A,1}^{(i)}, W_{A,2}^{(i)})$  as  $\mathsf{mha}^{(i)}(\cdot)$  and  $\mathsf{ffn}^{(i)}(\cdot)$  in the following. The output logits of the transformer is

$$X^{(L+1)} = X^{(L)} W_{\text{unemb}},$$

where  $W_{\text{unemb}} \in \mathbb{R}^{d \times d_{\text{vocab}}}$  is the unembedding matrix. We would like to adopt the last row of  $X^{(L+1)}$  to decode the next token. The parameters of the whole transformer is denoted as  $\theta = \{W_{Q,h}^{(i)}, W_{K,h}^{(i)}, W_{V,h}^{(i)}\}_{i,h=1}^{L,H} \cup \{W_{A,1}^{(i)}, W_{A,2}^{(i)}\}_{i=1}^{L} \cup \{W_{\text{unemb}}\}$ . Then the whole transformer is denoted as

$$X^{(L+1)} = \mathsf{transformer}(X, \theta).$$

In our method, we will apply a mask on the MHA in some layers, where we only remain the first and last several tokens. This can be described by defined the masked indexes set  $\mathcal{M}_i \subseteq [i]$  for *i*-row for  $i \in [N]$ . The corresponding mask  $M_{\text{lazy}}$  can be defined as

$$[M_{\mathsf{lazy}}]_{i,j} = \begin{cases} 0 & \text{if } j \notin \mathcal{M}_i \\ -\infty & \text{otherwise.} \end{cases}$$

For example, in our experiments, we set  $\mathcal{M}_i$  as the first 4 and the last 1020 tokens. Then we denote the corresponding MHA as

$$\widetilde{\mathsf{mha}}\Big(X,\{W_{Q,h},W_{K,h},W_{V,h}\}_{h=1}^H\Big) = \sum_{h=1}^H \mathsf{softmax}\big(XW_{Q,h}W_{K,h}^\top X^\top + M_{\mathsf{lazy}}\big)XW_{V,h}.$$

The  $\widetilde{\mathsf{mha}}$  module at *i*-th layer will be denoted as  $\widetilde{\mathsf{mha}}^{(i)}$ . The FF module will remain the same in the our method. We denote the set of indexes of the layers that apply this mask as  $\mathcal{I}$ . Then our method can be expressed as

$$\tilde{Y}^{(i)} = \tilde{X}^{(i-1)} + \mathbb{I}\{i \notin \mathcal{I}\} \cdot \mathsf{mha}^{(i)}\big(\mathsf{LN}(\tilde{X}^{(i-1)})\big) + \mathbb{I}\{i \in \mathcal{I}\} \cdot \widetilde{\mathsf{mha}}^{(i)}\big(\mathsf{LN}(\tilde{X}^{(i-1)})\big),$$

where we denote all the hidden states with our method applied as  $\tilde{X}$  and  $\tilde{Y}$ , and  $\mathbb{I}\{\cdot\}$  is the indicator function. The output of the whole network is denoted

$$\tilde{X}^{(L+1)} = \widetilde{\operatorname{transformer}}(X, \theta, \mathcal{I}).$$

To derive the theoretical analysis of the error, we need to delineate the norm of the transformer parameters. In fact, all the transformers in the real life have bounded parameters due to the calculation and storage requirements of the computer.

<span id="page-22-0"></span>**Assumption F.2.** The Frobenius norms of all the parameters of the transformer is upper bounded by B>0, i.e.,  $\|W_{Q,h}^{(i)}\|_{\mathbf{F}}\leq B$ ,  $\|W_{K,h}^{(i)}\|_{\mathbf{F}}\leq B$ ,  $\|W_{V,h}^{(i)}\|_{\mathbf{F}}\leq B$ ,  $\|W_{A,2}^{(i)}\|_{\mathbf{F}}\leq B$ ,  $\|W_{A,1}^{(i)}\|_{\mathbf{F}}\leq B$ ,  $\|W_{unemb}^{(i)}\|_{\mathbf{F}}\leq B$  for  $h\in[H]$  and  $i\in[L]$ .

To state our main result, we define the maximal sum of the original attention scores of the discarded tokens at layer  $l \in \mathcal{I}$  as  $s_l$ , which is formally defined as

$$\begin{split} s_l &= \max_{i \in [N]} \frac{1}{H} \sum_{h=1}^{H} \left( 1 - \sum_{j \notin \mathcal{M}_i} \frac{\exp\left( \left[ \mathsf{LN}(X^{(l-1)}) \right]_{i,:} W_{Q,h}^{(i)} W_{K,h}^{(i),\top} \left[ \mathsf{LN}(X^{(l-1)})^\top \right]_{:,j} \right)}{\sum_{k=1}^{i} \exp\left( \left[ \mathsf{LN}(X^{(l-1)}) \right]_{i,:} W_{Q,h}^{(i)} W_{K,h}^{(i),\top} \left[ \mathsf{LN}(X^{(l-1)})^\top \right]_{:,k} \right)} \right) \\ &= \max_{i \in [N]} \frac{1}{H} \sum_{h=1}^{H} \sum_{j \in \mathcal{M}_i} \frac{\exp\left( \left[ \mathsf{LN}(X^{(l-1)}) \right]_{i,:} W_{Q,h}^{(i)} W_{K,h}^{(i),\top} \left[ \mathsf{LN}(X^{(l-1)})^\top \right]_{:,j} \right)}{\sum_{k=1}^{i} \exp\left( \left[ \mathsf{LN}(X^{(l-1)}) \right]_{i,:} W_{Q,h}^{(i)} W_{K,h}^{(i),\top} \left[ \mathsf{LN}(X^{(l-1)})^\top \right]_{:,k} \right)}. \end{split}$$

Then the main result is as follows.

<span id="page-22-1"></span>**Theorem F.3.** We define the difference of the hidden states of our method and the original transformer at layer  $i \in [L]$  as  $e_X^{(i)} = \|X^{(i)} - \tilde{X}^{(i)}\|_{2,\infty}$ . Under Assumptions F.1 and F.2, this error involves as

$$e_X^{(i)} \leq e_X^{(i-1)} + \left(HB + L_{\mathsf{lip}}B^2 + 4HB^3\right) \min\left\{2, \left[1 + HB(1 + 4B^2)\right]e_X^{(i-1)}\right\} + 2H(B + L_{\mathsf{lip}}B^3)\mathbb{I}\{i \in \mathcal{I}\}s_i. \tag{4}$$

The error between the logits generated by our method and the original transformer can be upper-bounded as

$$\left\| \widetilde{\operatorname{transformer}}(X, \theta, \mathcal{I}) - \operatorname{transformer}(X, \theta) \right\|_{2, \infty} \leq 2LB^2 \left( H + L_{\operatorname{lip}}B + 4HB^2 \right) + 2HB^2 (1 + L_{\operatorname{lip}}B^2) \sum_{i \in \mathcal{I}} s_i. \tag{5}$$

We note that the error recursive expression consists of three terms. The first term represents the error from the previous layer. The second term represents the error from the previous layer amplified by the current layer. Thanks to the layer normalization, this term will be truncated by 2. The last term represents the newly introduced error if we shorten KV cache at the current layer. By relaxing this recursive formula, we derive the upper bound of the error between logits of our method and the original transformer. This shows that the error is upper bounded by the sum of the attention scores of the removed KV pairs up to an additive constant.

Proof of Theorem F.3. We derive the error analysis of our analysis in three steps.

- The error decomposition of the whole network.
- Bound each term in the error decomposition.
- Conclude the proof.

#### Step 1: The error decomposition of the whole network.

We derive the error decomposition of the whole network in a recursive manner. In fact, for the i-th layer, we have that

$$\begin{split} \|\tilde{X}^{(i)} - X^{(i)}\|_{2,\infty} &\leq \|\tilde{Y}^{(i)} - Y^{(i)}\|_{2,\infty} + \left\| \mathsf{ffn}^{(i)} \big( \mathsf{LN}(\tilde{Y}^{(i)}) \big) - \mathsf{ffn}^{(i)} \big( \mathsf{LN}(Y^{(i)}) \big) \right\|_{2,\infty} \\ \|\tilde{Y}^{(i)} - Y^{(i)}\|_{2,\infty} &\leq \|\tilde{X}^{(i-1)} - X^{(i-1)}\|_{2,\infty} \\ &+ \left\| \mathbb{I}\{i \notin \mathcal{I}\} \cdot \mathsf{mha}^{(i)} \big( \mathsf{LN}(\tilde{X}^{(i-1)}) \big) + \mathbb{I}\{i \in \mathcal{I}\} \cdot \widetilde{\mathsf{mha}}^{(i)} \big( \mathsf{LN}(\tilde{X}^{(i-1)}) \big) - \mathsf{mha}^{(i)} \big( \mathsf{LN}(X^{(i-1)}) \big) \right\|_{2,\infty}, \end{split}$$

where the inequalities follow from the triangle inequality. In addition, we have that

$$\left\| \widetilde{\operatorname{transformer}}(X, \theta, \mathcal{I}) - \operatorname{transformer}(X, \theta) \right\|_{2, \infty} \le \|W_{\operatorname{unemb}}\|_{\mathbf{F}} \cdot \|X^{(L)} - \tilde{X}^{(L)}\|_{2, \infty}, \tag{8}$$

where the inequality results from Lemma G.2.

#### Step 2: Bound each term in the error decomposition

We will bound each term in the right-hand side of Eqn. equation 6 and equation 7. For the term related to the FF module, we have that

<span id="page-23-5"></span><span id="page-23-2"></span><span id="page-23-1"></span><span id="page-23-0"></span>
$$\begin{split} & \left\| \mathsf{ffn}^{(i)} \big( \mathsf{LN}(\tilde{Y}^{(i)}) \big) - \mathsf{ffn}^{(i)} \big( \mathsf{LN}(Y^{(i)}) \big) \right\|_{2,\infty} \\ & \leq L_{\mathsf{lip}} \cdot \|W_{A,2}^{(i)}\|_{\mathbf{F}} \cdot \|W_{A,1}^{(i)}\|_{\mathbf{F}} \cdot \|\mathsf{LN}(\tilde{Y}^{(i)}) - \mathsf{LN}(Y^{(i)}) \|_{2,\infty} \\ & \leq L_{\mathsf{lip}} \cdot \|W_{A,2}^{(i)}\|_{\mathbf{F}} \cdot \|W_{A,1}^{(i)}\|_{\mathbf{F}} \cdot \min \left\{ 2, \|\tilde{Y}^{(i)} - Y^{(i)}\|_{2,\infty} \right\} \\ & \leq L_{\mathsf{lip}} \cdot B^{2} \cdot \min \left\{ 2, \|\tilde{Y}^{(i)} - Y^{(i)}\|_{2,\infty} \right\}, \end{split} \tag{9}$$

where the first inequality results from Lemma G.2, the second inequality results from the definition of  $ln(\cdot)$ , and the last inequality results from Assumption F.2. For the term related to MHA module in the right-hand side of Eqn. equation 7, we have that

$$\begin{split} \left\| \mathbb{I}\{i \notin \mathcal{I}\} \cdot \mathsf{mha}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) + \mathbb{I}\{i \in \mathcal{I}\} \cdot \widetilde{\mathsf{mha}}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) - \mathsf{mha}^{(i)} \left( \mathsf{LN}(X^{(i-1)}) \right) \right\|_{2,\infty} \\ &= \mathbb{I}\{i \notin \mathcal{I}\} \cdot \left\| \mathsf{mha}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) - \mathsf{mha}^{(i)} \left( \mathsf{LN}(X^{(i-1)}) \right) \right\|_{2,\infty} \\ &+ \mathbb{I}\{i \in \mathcal{I}\} \cdot \left\| \widetilde{\mathsf{mha}}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) - \mathsf{mha}^{(i)} \left( \mathsf{LN}(X^{(i-1)}) \right) \right\|_{2,\infty} \\ &\leq \mathbb{I}\{i \notin \mathcal{I}\} \cdot \left\| \mathsf{mha}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) - \mathsf{mha}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) \right\|_{2,\infty} \\ &+ \mathbb{I}\{i \in \mathcal{I}\} \cdot \left( \left\| \widetilde{\mathsf{mha}}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) - \mathsf{mha}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) \right\|_{2,\infty} \right. \\ &+ \left\| \mathsf{mha}^{(i)} \left( \mathsf{LN}(\tilde{X}^{(i-1)}) \right) - \mathsf{mha}^{(i)} \left( \mathsf{LN}(X^{(i-1)}) \right) \right\|_{2,\infty} \right. \\ &\leq H \cdot B \left( 1 + 4B^2 \right) \left\| \mathsf{LN}(X^{(i-1)}) - \mathsf{LN}(\tilde{X}^{(i-1)}) \right\|_{2,\infty} + \mathbb{I}\{i \in \mathcal{I}\} \cdot 2BH \cdot s_i \\ &\leq H \cdot B \left( 1 + 4B^2 \right) \min \left\{ 2, \left\| X^{(i-1)} - \tilde{X}^{(i-1)} \right\|_{2,\infty} \right\} + \mathbb{I}\{i \in \mathcal{I}\} \cdot 2BH \cdot s_i, \end{split} \tag{10}$$

where the first inequality results from the triangle inequality, the second inequality results from Lemma G.4. Define the error  $e_X^{(i)} = \|X^{(i)} - \tilde{X}^{(i)}\|_{2,\infty}$  with  $e_X^{(0)} = 0$ . Combining Eqn. equation 6, equation 7, equation 9, and equation 10, we have that

<span id="page-23-4"></span><span id="page-23-3"></span>
$$e_X^{(i)} \le e_X^{(i-1)} + HB(1+4B^2) \min\{2, e_X^{(i-1)}\} + \mathbb{I}\{i \in \mathcal{I}\}2BHs_i + L_{\text{lip}}B^2 \min\{2, e_X^{(i-1)} + HB(1+4B^2) \min\{2, e_X^{(i-1)}\} + \mathbb{I}\{i \in \mathcal{I}\}2BHs_i\}.$$
(11)

#### Step 3: Conclude the proof.

We derive the recursive expression of the hidden state error by relaxing the right-hand side of Eqn. equation 11 as follows.

$$\begin{split} e_X^{(i)} &\leq e_X^{(i-1)} + HB(1+4B^2) \min\{2, e_X^{(i-1)}\} + \mathbb{I}\{i \in \mathcal{I}\}2BHs_i \\ &\quad + L_{\mathsf{lip}}B^2 \min\left\{2, \left[1 + HB(1+4B^2)\right]e_X^{(i-1)}\right\} + \mathbb{I}\{i \in \mathcal{I}\}2L_{\mathsf{lip}}B^3Hs_i \\ &\leq e_X^{(i-1)} + \left(HB + L_{\mathsf{lip}}B^2 + 4HB^3\right) \min\left\{2, \left[1 + HB(1+4B^2)\right]e_X^{(i-1)}\right\} + 2H(B + L_{\mathsf{lip}}B^3)\mathbb{I}\{i \in \mathcal{I}\}s_i. \end{split}$$

This proves the recursive formula. By summing this inequality from i = 1 to i = L, we have that

$$e_X^{(L)} \le 2L(HB + L_{\mathsf{lip}}B^2 + 4HB^3) + 2(B + L_{\mathsf{lip}}B^3) \sum_{i \in \mathcal{I}} s_i.$$
 (12)

<span id="page-24-2"></span>

Combining Eqn. equation 8 and equation 12, we have that

$$\left\| \widetilde{\operatorname{transformer}}(X,\theta,\mathcal{I}) - \operatorname{transformer}(X,\theta) \right\|_{2,\infty} \leq 2LB^2 \left( H + L_{\operatorname{lip}}B + 4HB^2 \right) + 2B^2 H (1 + L_{\operatorname{lip}}B^2) \sum_{i \in \mathcal{I}} s_i.$$

Thus, we conclude the proof of Theorem F.3.

## G Supporting Lemmas

**Lemma G.1** (Corollary A.7 in Edelman et al. (2022) ). For any  $x, y \in \mathbb{R}^d$ , we have

$$\|\operatorname{softmax}(x) - \operatorname{softmax}(y)\|_1 \le 2\|x - y\|_{\infty}.$$

<span id="page-24-0"></span>**Lemma G.2** (Lemma 17 in Zhang et al. (2022) ). Given any two conjugate numbers  $u, v \in [1, \infty]$ , i.e.,  $\frac{1}{u} + \frac{1}{v} = 1$ , and  $1 \le p \le \infty$ , for any  $A \in \mathbb{R}^{r \times c}$  and  $x \in \mathbb{R}^c$ , we have

$$||Ax||_p \le ||A^\top||_{p,u} ||x||_v \quad and \quad ||Ax||_p \le ||A||_{u,p} ||x||_v.$$

**Lemma G.3** (Lemma I.8 in Zhang et al. (2023)). For any  $X, \tilde{X} \in \mathbb{R}^{N \times d}$ , and any  $W_{Q,h}, W_{K,h} \in \mathbb{R}^{d \times d_h}, W_{V,h} \in \mathbb{R}^{d \times d}$  for  $h \in [H]$ , if  $\|X\|_{2,\infty}, \|\tilde{X}\|_{2,\infty} \leq B_X$ ,  $\|W_{Q,h}\|_{\mathbf{F}} \leq B_Q$ ,  $\|W_{K,h}\|_{\mathbf{F}}, \leq B_K$ ,  $\|W_{V,h}\|_{\mathbf{F}} \leq B_V$  for  $h \in [H]$ , then we have

$$\begin{split} & \left\| \mathsf{mha} \big( X, \{ W_{Q,h}, W_{K,h}, W_{V,h} \}_{h=1}^H \big) - \mathsf{mha} \big( \tilde{X}, \{ W_{Q,h}, W_{K,h}, W_{V,h} \}_{h=1}^H \big) \right\|_{2,\infty} \\ & \leq H \cdot B_V \big( 1 + 4 B_X^2 \cdot B_Q B_K \big) \| X - \tilde{X} \|_{2,\infty}. \end{split}$$

<span id="page-24-1"></span>**Lemma G.4.** For a query vector  $q \in \mathbb{R}^d$ , and two sets of key-value pairs  $K_1 \in \mathbb{R}^{N_1 \times d}$ ,  $K_2 \in \mathbb{R}^{N_2 \times d}$ ,  $V_1 \in \mathbb{R}^{N_1 \times d}$ , and  $V_2 \in \mathbb{R}^{N_2 \times d}$ , We define attention scores softmax $(q^{\top}[K_1, K_2]^{\top})$  and softmax $(q^{\top}K_1^{\top})$  as

$$\operatorname{softmax}(q^{\top}[K_1,K_2]^{\top}) = [s_1^{\top},s_2^{\top}], \ \ and \ \operatorname{softmax}(q^{\top}K_1^{\top}) = \tilde{s}_1^{\top}.$$

Then we have that

$$\left\| \mathsf{softmax}(q^\top K_1^\top) V_1 - \mathsf{softmax}(q^\top [K_1, K_2]^\top) [V_1^\top, V_2^\top]^\top \right\|_2 \leq 2 \|s_2\|_1 \cdot \max\{\|V_1\|_{2,\infty}, \|V_2\|_{2,\infty}\}.$$

Proof of Lemma G.4. In fact, we have that

$$\mathsf{softmax}(q^{\top}[K_1,K_2]^{\top})[V_1^{\top},V_2^{\top}]^{\top} = s_1^{\top}V_1 + s_2^{\top}V_2, \text{ and } \mathsf{softmax}(q^{\top}K_1^{\top})V_1 = \tilde{s}_1^{\top}V_1.$$

Further, the difference between  $s_1$  and  $\tilde{s}_1$  can be upper bounded as

$$\begin{split} &\|s_1 - \tilde{s}_1\|_1 \\ &= \sum_{i=1}^{N_1} \left| \frac{\exp(q^\top [K_1]_{i,:})}{\sum_{j=1}^{N_1} \exp(q^\top [K_1]_{j,:}) + \sum_{l=1}^{N_2} \exp(q^\top [K_2]_{l,:})} - \frac{\exp(q^\top [K_1]_{i,:})}{\sum_{j=1}^{N_1} \exp(q^\top [K_1]_{j,:})} \right| \\ &= \sum_{i=1}^{N_1} \frac{\exp(q^\top [K_1]_{i,:}) \sum_{l=1}^{N_2} \exp(q^\top [K_2]_{l,:})}{\left(\sum_{j=1}^{N_1} \exp(q^\top [K_1]_{j,:}) + \sum_{l=1}^{N_2} \exp(q^\top [K_2]_{l,:})\right) \sum_{j=1}^{N_1} \exp(q^\top [K_1]_{j,:})} \\ &= \|s_2\|_1, \end{split}$$

where the first equality results from the definition of  $softmax(\cdot)$ , and the last equality results from the definition of  $s_2$ . Then we have that

$$\begin{split} & \left\| \operatorname{softmax}(q^\top K_1^\top) V_1 - \operatorname{softmax}(q^\top [K_1, K_2]^\top) [V_1^\top, V_2^\top]^\top \right\|_2 \\ & = \left\| s_1^\top V_1 + s_2^\top V_2 - \tilde{s}_1^\top V_1 \right\|_2 \\ & \leq \| s_1 - \tilde{s}_1 \|_1 \cdot \| V_1 \|_{2,\infty} + \| s_2 \|_1 \cdot \| V_2 \|_{2,\infty} \\ & \leq 2 \| s_2 \|_1 \cdot \max\{ \| V_1 \|_{2,\infty}, \| V_2 \|_{2,\infty} \}. \end{split}$$

Thus, we conclude the proof of Lemma G.4.