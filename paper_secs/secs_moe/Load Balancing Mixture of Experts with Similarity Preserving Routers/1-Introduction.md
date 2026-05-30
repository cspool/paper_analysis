# 1 Introduction

As the demand for larger and more capable neural networks continues to grow [\[Kaplan et al., 2020,](#page-10-0) [Brown et al., 2020\]](#page-9-0), the need for architectures that can scale efficiently—without incurring prohibitive computational costs—has become increasingly important. This is especially true in the context of large language models (LLMs), where state-of-the-art performance often requires billions of parameters and massive training datasets. One such approach, the Mixture of Experts (MoE) model [\[Shazeer et al., 2017\]](#page-11-0), introduces sparsely activated sub-networks at certain layers, allowing for increased model capacity while preserving computational efficiency.

While MoE architectures offer improved parameter scalability, they often suffer from poor expert utilization during pretraining. Without mechanisms that encourage balanced routing, the model frequently learns to rely on only a small subset of experts [\[Eigen et al., 2014,](#page-10-1) [Bengio et al., 2016\]](#page-9-1). Typically, routing decisions are made per token using a learned router that outputs a probability distribution over experts—a paradigm known as Token Choice (TC) [\[Fedus et al., 2022\]](#page-10-2). To encourage balanced expert usage, various strategies have been proposed, including sequence-level auxiliary losses such as load balancing loss (LBL) [\[Fedus et al., 2022\]](#page-10-2) or the Expert Choice (EC) routing variant which generates a distribution over a sparse set of activated tokens for each expert [\[Zhou et al.,](#page-12-0) [2022\]](#page-12-0). Section [5](#page-8-0) covers additional strategies for load balancing.

Load balancing strategies often encourage a uniform distribution over experts to avoid collapse. This approach has proven to be useful to stabilize MoEs during training, and has been used in many recent works [\[Muennighoff et al., 2025,](#page-11-1) [Dai et al., 2024,](#page-10-3) [DeepSeek-AI et al., 2025,](#page-10-4) [Xue et al., 2024\]](#page-12-1). However, in this paper, we argue that imposing a uniform distribution over experts causes MoE models to expend their capacity acquiring the same knowledge across multiple experts. Besides

<sup>˚</sup>Corresponding author: nabilomi@cs.uw.edu

the inefficiencies imposed by this approach, exposing similar tokens to several different experts during training results in inconsistent routing behavior and expert assignments. This in turn further exacerbates knowledge redundancy across experts. Previous work [Dai et al., 2024, Liu et al., 2024] suggests that the amount of knowledge shared between experts is correlated to losses in performance.

To encourage consistent expert assignments for similar input tokens during training, we propose preserving the relational structure among tokens during routing, resulting in similar expert distributions for similar tokens. We achieve this by promoting orthogonality in the router's weights, as orthogonal matrices are dot-product (and thus, angle) preserving. We introduce **sim**ilarity-preserving routers for MoE load **balancing** (SIMBAL), a novel load balancing auxiliary loss that maintains token-wise relational structure by softly encouraging orthogonality in the router weights. Unlike methods that impose orthogonality through explicit parameter constraints—which are computationally expensive and numerically unstable (see Section 4.1)—SIMBAL aligns the Gram matrix  $(Q^TQ)$  of router weights with the identity matrix. This softly regularizes router outputs to preserve pairwise token similarities, achieving the benefits of orthogonal routing with significantly lower computational cost.

By maintaining semantic structure and promoting diverse expert usage, SIMBAL reduces redundancy, accelerates convergence, and improves final model quality. Our models require 36% fewer tokens when training to achieve the same loss as LBL, and achieve 0.213 lower perplexity given the same compute budget.

#### 2 Background

#### 2.1 Mixtures of Experts

A Mixture of Experts (MoE) model *sparsely activates* certain parameters during inference, in contrast to standard dense networks where all parameters are used. In this work, we focus on Mixture of Experts models for the Transformer architecture [Vaswani et al., 2017], a popular choice for training models on sequence-wise data such as those seen in natural language.

Transformers are typically composed of a series of blocks, each consisting of a self-attention module followed by a feed-forward network (FFN). The FFN is usually a two-layer fully connected network with a large hidden dimensionality. For example, given an input vector  $x \in \mathbb{R}^{D_M}$ , where  $D_M$  is the model (input/output) dimensionality, the standard FFN computes:

$$FFN(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2, \tag{1}$$

where  $W_1 \in \mathbb{R}^{D_F \times D_M}$ ,  $W_2 \in \mathbb{R}^{D_M \times D_F}$ ,  $b_1 \in \mathbb{R}^{D_F}$ , and  $b_2 \in \mathbb{R}^{D_M}$ . The intermediate hidden dimension  $D_F$  is typically much larger than  $D_M$ . The nonlinearity  $\sigma$  is an activation function; we use SwiGLU [Shazeer, 2020].

In a Mixture of Experts Transformer, the FFN is replaced by a set of smaller, parallel FFNs called "experts." Let there be E such experts. Each expert has its own parameters  $\{W_1^{(e)},W_2^{(e)},b_1^{(e)},b_2^{(e)}\}$ , where  $W_1^{(e)} \in \mathbb{R}^{D_E \times D_M},W_2^{(e)} \in \mathbb{R}^{D_M \times D_E},b_1^{(e)} \in \mathbb{R}^{D_E}$ , and  $b_2^{(e)} \in \mathbb{R}^{D_M}$ . Here,  $D_E$  is the hidden dimension used within each expert.

A routing mechanism assigns each token  $x \in \mathbb{R}^{D_M}$  to a small subset of A activated experts (typically  $A \ll E$ ). The router is a linear transformation  $R \in \mathbb{R}^{D_M \times E}$  followed by a sparse top-A selection, producing expert indices  $i_1, \ldots, i_A$  and associated routing weights  $r_1, \ldots, r_A$ . The MoE layer then computes:

$$MoE(x) = \sum_{a=1}^{A} r_a \cdot \left( W_2^{(i_a)} \cdot \sigma(W_1^{(i_a)} x + b_1^{(i_a)}) + b_2^{(i_a)} \right).$$
 (2)

This definition of the MoE can also be viewed as a weighted sum over expert FFN outputs, skipping the computation for any expert where the weight is zero. This architecture enables scaling model capacity via E without a proportional increase in computational cost, as only A experts are active per input.

#### 2.2 Expert Routing

Despite the small parameter count of MoE routers (in our larger setting, 0.018% of the total parameters), they have an outsized impact on the performance and capacity of the model, as they orchestrate billions of parameters. Thus, it is imperative to pay careful attention to this mechanism when training MoE models. In MoE Transformers, routing is computed from the previous attention output  $x \in \mathbb{R}^{D_M}$  via a learned router matrix  $R \in \mathbb{R}^{D_M \times E}$ , producing scores  $xR \in \mathbb{R}^E$ . Applying a gating function G results in routing weights r = G(xR). We use softmax, which generates a probability distribution over experts, from which the top-A active experts are selected and weighted for each token.

We compare our approach to balancing with the Load Balancing Loss (LBL) presented by Fedus et al. [2022]. This setup is highly popular and represents the state-of-the-art, being used in Muennighoff et al. [2025], DeepSeek-AI et al. [2025], Dai et al. [2024], and [Xue et al., 2024] (we give an overview of alternative methods and their limitations in Section 5.) LBL encourages uniform expert usage by correlating how frequently each expert is selected with how much routing weight it receives. Let  $f_i$  be the fraction of tokens routed to expert i,  $P_i$  the average routing probability for expert i, and E the number of experts. The LBL is defined as:

$$\mathcal{L}_{LBL} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i \tag{3}$$

Minimizing this loss encourages the router to distribute tokens more evenly across experts. However, it may require tuning of a loss coefficient  $\alpha$  to avoid overpowering the main training objective. We include PyTorch implementation details in Appendix A.3.

#### 3 Methods

We propose preserving token-wise structural relationships to ensure effective and consistent usage of experts during training. We accomplish this by encouraging orthogonality in the router, which preserves the pairwise angles of the inputs. In this section, we explain the methods used to achieve our results, and our design choices.

#### <span id="page-2-0"></span>3.1 Load Balancing via Orthonormal Routers

A natural strategy to ensure expert choices correlate with token-wise relationships is to constrain the router weights to form an orthonormal (and thus, dot-product preserving) matrix. PyTorch [Paszke et al., 2019] provides a utility for this using a QR decomposition, producing a matrix  $Q \in \mathbb{R}^{m \times n}$  such that  $Q^{\top}Q = I_n$  if  $m \geqslant n$  (as is typically the case with MoE routers).

While appealing, the cost of using this orthogonal parameterization is prohibitively expensive in wall-clock time when applied to large-scale models, because the algorithms used to ensure this property are computationally expensive. Instead, we propose a loss that encourages structure preservation without requiring explicit parameterization.

Let the router be a matrix  $R \in \mathbb{R}^{D_M \times E}$ , where  $D_M$  is the model dimension and E is the number of experts. Since  $E \ll D_M$ , we minimize the deviation of the Gram matrix  $R^{\top}R$  from the identity:

$$\mathcal{L}_{\text{orth}} = \|R^{\top}R - I_E\|_1 \tag{4}$$

This loss is dataset-agnostic and computationally cheap. This is important, as Qiu et al. [2025] finds that existing losses, which are dependent on the data, require large batch sizes to be effective. We additionally initialize the router with a (near) orthogonal initialization [Saxe et al., 2014] (though it should be sufficient to simply run a few router-only training steps, see Table 2), as we find it results in quicker convergence. We call this method SIMBAL, as we are effectively balancing by preserving the pair-wise similarity of the tokens. The experiments in our paper scale this coefficient by 0.1, but we find that this is not important, as shown in Section 4.2. We include PyTorch implementation details in Appendix A.3.

<span id="page-3-0"></span>Table 1: Parameters used for the model architecture and training. Parameter (active, total) counts include token embeddings. All MoE models have 32 experts, with the top 4 activated.

| Parameter        | Dense-M | MoE-M | Dense-L | MoE-L |
|------------------|---------|-------|---------|-------|
| DM               | 768     | 768   | 1536    | 1536  |
| Depth            | 8       | 8     | 12      | 12    |
| Heads            | 8       | 8     | 12      | 12    |
| DF               | 3072    | 768   | 6144    | 1536  |
| RoPE θ           | 1e4     | 1e4   | 1e5     | 1e5   |
| Peak LR          | 5e-4    | 5e-4  | 3e-4    | 3e-4  |
| Embedding Params | 77M     | 77M   | 154M    | 154M  |
| Active Params    | 230M    | 230M  | 761M    | 761M  |
| Total Params     | 230M    | 627M  | 761M    | 3.14B |

## <span id="page-3-1"></span>3.2 Model Architecture and Training

Model Architecture. Our model architecture closely follows prior work by [OLMo et al.](#page-11-7) [\[2025\]](#page-11-7) and [Muennighoff et al.](#page-11-1) [\[2025\]](#page-11-1). We use a Transformer backbone with RMSNorm [\[Zhang and Sennrich,](#page-12-3) [2019\]](#page-12-3), SwiGLU activations [\[Shazeer, 2020\]](#page-11-3), and Rotary Position Embeddings (RoPE) [\[Su et al.,](#page-11-8) [2021\]](#page-11-8). We apply Z-loss [Team](#page-12-4) [\[2025\]](#page-12-4), [Chowdhery et al.](#page-9-2) [\[2022\]](#page-9-2) with a coefficient of 1e-5, as in [OLMo et al.](#page-11-7) [\[2025\]](#page-11-7). Unlike OLMo 2, we do not modify the placement of normalization layers nor do we apply QK-Norm [\[Dehghani et al., 2023\]](#page-10-5). We replace all FFN layers with MoE layers. Further architectural details can be found in Table [1.](#page-3-0) Our implementation builds upon the open-source OLMo codebase [\[OLMo et al., 2025\]](#page-11-7).

For the LBL baseline, we follow [Muennighoff et al.](#page-11-1) [\[2025\]](#page-11-1) and [Wang et al.](#page-12-5) [\[2024\]](#page-12-5), using a loss coefficient of 0.01. In contrast, our method does not require a coefficient; with appropriate initialization, the load-balancing loss converges quickly.

Model Scales and Training. We pretrain models at two scales: a medium model (MoE-M) with 230M active and 627M total parameters, and a large model (MoE-L) with 762M active and 3.14B total parameters (including embeddings). For each scale, we performed a brief hyperparameter sweep across three learning rates. All models are trained using the AdamW optimizer [\[Loshchilov and](#page-11-9) [Hutter, 2019\]](#page-11-9), with a weight decay of 0.01, linear warm-up from 10% of the peak learning rate over 2000 steps, followed by cosine decay [\[Loshchilov and Hutter, 2017\]](#page-11-10) to 10% of the peak learning rate. Additional model specifications are listed in Table [1.](#page-3-0) All model parameters are in bfloat16.

All models are trained on a subset of tokens from the DCLM-pool-400m-1x dataset [\[Li et al., 2025\]](#page-10-6) (used in other work such as [Muennighoff et al.](#page-11-1) [\[2025\]](#page-11-1)), tokenized using the cl100k\_base tokenizer from the tiktoken library [\[OpenAI, 2024\]](#page-11-11). We reserve one file shard (77M tokens) for validation. All MoE-M models are trained on 19.9B tokens, while MoE-L mdoels are trained on 78.6B tokens. No further fine-tuning is performed, as our focus is on the pretraining phase, which is typically the most computationally intensive stage of LLM development.

Compute and FLOP Estimates. All models are trained using Distributed Data Parallelism (DDP) [\[Li et al., 2020\]](#page-11-12). For MoE-M, we use 8 NVIDIA A100 40GB GPUs per training run; for MoE-L, we use 8 AMD MI300X 192GB accelerators.

To estimate total training FLOPs, we follow the approximation from [Brown et al.](#page-9-0) [\[2020\]](#page-9-0), using 6 ˆ N ˆ T per forward pass, where N is the number of non-embedding active parameters and T is the number of training tokens.

For MoE-M and Dense-M, with 230M active parameters and 77M in embeddings, trained on 2ˆ10<sup>10</sup> tokens, this results in:

$$6 \times ((230 - 77) \times 10^6) \times 2 \times 10^{10} = 1.836 \times 10^{19} \text{ FLOPs}$$

For MoE-L and Dense-L, with 761M active parameters and 154M in embeddings, trained on 7.8 ˆ 10<sup>10</sup> tokens, this results in:

$$6 \times ((761 - 154) \times 10^6) \times 7.8 \times 10^{10} = 2.840 \times 10^{20} \text{ FLOPs}$$

<span id="page-4-1"></span>Table 2: Comparison of orthogonality preservation methods, average and standard deviation over 100 trials. We report the maximum deviation from orthonormality (**Max Dev**) and the mean L1 distance to the identity matrix (**L1 Dist**) after casting to our training precision. **Trained** refers to our loss-based method after 100 optimization steps. **Param** uses the orthogonal parameterization from Lezcano-Casado [2019]. **OrthoInit** follows the initialization from Saxe et al. [2014]. All matrices have shape  $1536 \times 32$ , matching our router dimensions. Best results in each column are bolded.

| Method    | Max Dev                                       | L1 Dist                                       |
|-----------|-----------------------------------------------|-----------------------------------------------|
| Trained   | $1.03 \times 10^{-5} \pm 2.76 \times 10^{-6}$ |                                               |
| Param     | $2.00 \times 10^{-4} \pm 2.31 \times 10^{-5}$ | $4.80 \times 10^{-5} \pm 1.60 \times 10^{-6}$ |
| OrthoInit | $1.93 \times 10^{-4} \pm 1.88 \times 10^{-5}$ | $4.62 \times 10^{-5} \pm 1.79 \times 10^{-6}$ |

#### <span id="page-4-2"></span>3.3 Measuring Expert Similarity

Previous work evaluates expert specialization by measuring performance degradation when the top fraction of experts is dropped [Dai et al., 2024]. However, this approach is expensive to compute when ablating each combination of dropped experts for exhaustive comparison, as it requires inference on the full validation set per combination of dropped experts.

We instead propose *Pairwise Expert Similarity (PES)*: a smoother, scalable, and robust metric for quantifying expert specialization based on the similarity of expert outputs across a batch of tokens. Ideally, specialized experts should produce more diverse (i.e., less similar) outputs, maximizing the representational span of the expert set. PES is defined as:

$$PES_{model} = \frac{1}{|B|} \sum_{b \in B} C_{expert}(\mathbf{x})$$
 (5)

$$C_{\text{expert}}(\mathbf{x}) = \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \cos\left(\mathbf{f}_{i}(\mathbf{x}), \mathbf{f}_{j}(\mathbf{x})\right)$$
(6)

Here,  $C_{\text{expert}}(\mathbf{x})$  denotes the mean cosine similarity of expert outputs for batch sample  $\mathbf{x}$ , and PES<sub>model</sub> is the batch-averaged similarity across all |B| samples. N is the number of experts,  $\mathbf{f}_i$  is the function computed by the i-th expert. The cosine similarity  $\cos(\mathbf{u}, \mathbf{v})$  is defined as  $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$ , measuring the angle between output vectors.

PES is intuitive (lower similarity indicates greater diversity and lower redundancy), considers all experts rather than just the most frequently selected, and is highly scalable. Unlike dropout-based evaluation, which requires repeated forward passes per ablation, PES requires less additional computation. This function can be computed batch-wise within the expert computation to reduce cost, and requires inference once with the full model parameter count (a multiplier of 3.6-4.9x FLOPs in our case), rather than (potentially) hundreds of evaluation passes with the MoE for similarly comprehensive evaluations. We use 4M randomly sampled tokens to calculate PES.

## 4 Experiments

#### <span id="page-4-0"></span>4.1 Orthogonalization and Balancing

Our key contribution is that we perform load balancing by using a router that is encouraged to be orthogonal, and thus preserves token-wise relationships. Rather than enforcing orthogonality through explicit parameter constraints—which is computationally expensive, requires frequent reparameterization, and is prone to numerical instability, particularly when training large-scale models—we instead use the loss function described in Section 3.1. We now evaluate the effectiveness of promoting orthogonality in the router.

As PyTorch currently lacks support for orthogonal parameterizations in lower-precision formats commonly used to train language models (that we use), we perform orthogonalization in float32, and then cast the resulting matrix to bfloat16, our training precision. Our loss-based method

Table 3: Load balancing and orthogonalization of LBL and SIMBAL on MoE-L.

<span id="page-5-1"></span>

| Metric | SEU   | Entropy | $(R^TR - I)^2$         |
|--------|-------|---------|------------------------|
| LBL    | 1.000 | 1.268   | 0.0311                 |
| SIMBAL | 0.991 | 1.168   | $2.121 \times 10^{-8}$ |

![](_page_5_Figure_2.jpeg)

![](_page_5_Figure_3.jpeg)

<span id="page-5-2"></span>Figure 1: Validation loss curves for checkpoints during training. In both MoE-M and MoE-L, we achieve the same loss roughly 36% faster.

trains the matrix directly in bfloat16. We report both the maximum and mean deviation from orthonormality, as well as the final loss values, in Table 2. We find that our loss consistently produces matrices that more closely approximate orthonormality than direct orthogonal parameterizations in our scenario. In fact, our approach matches or exceeds the throughput of efficient orthogonal parameterizations, while avoiding the need for expensive reorthogonalization steps. For this synthetic experiment, we train with AdamW (with no weight decay), and a learning rate cosine decayed from  $1 \times 10^{-4}$  to  $1 \times 10^{-5}$  over 100 consecutive steps. In our MoEs, we simply add our loss as an auxiliary loss term and update once per language model training step. We examine the coefficient sensitivity of SIMBAL to determine if tuning is necessary.

In terms of expert utilization in MoEs, our method avoids collapse comparably to LBL, ensuring that no experts remain unutilized. Figure 2 illustrates the unique expert usage over time at two different scales, compared to LBL and using no losses (which results in unused experts). To verify that sequence-wise balance is not substantially degraded, we compare SIMBAL against LBL by measuring the entropy of the routing distributions and Sequence-wise Expert Utilization (SEU), as reported by the mean over the fraction of experts used per sequence, to show that load balance within a sequence is not significantly degraded. We report our results in Table 3.

To analyze whether SIMBAL is able to effectively orthogonalize routing matrices, we analyze the mean layer-wise L2 distance of the final router gram matrix from the identity matrix in Table 3. More in-depth data with layer-wise values across MoE-L and MoE-M can be found in Appendix A.2.

#### <span id="page-5-0"></span>4.2 Language Modeling

We compare our method to LBL by training language models according to the setup described in Section 3.2, evaluating performance based on the perplexity of the final checkpoint. The resulting models are reported in Table 4. We additionally report the SEU of the models.

Across both MoE-M and MoE-L scales, SimBal converges approximately 36% faster than LBL. We show validation values during training in Figure 1 For MoE-L, SimBal approaches the target loss after processing roughly 50B tokens, compared to 78.6B for LBL—a 36% improvement. Similarly, in the MoE-M setting, SimBal reaches comparable loss levels at around 12.7B tokens, versus 19.9B for LBL. We additionally evaluate MoE-L on standard downstream benchmarks to test whether the perplexity gains of SIMBAL translate to broader tasks, comparing against LBL (Table 5). Overall, our method outperforms LBL in both downstream performance and training efficiency.

We train 4 additional models (for a total of 5 models) for both SIMBAL and LBL on MoE-M (due to computational limitations) to parse the statistical significance of our results. We find that models

![](_page_6_Figure_0.jpeg)

![](_page_6_Figure_1.jpeg)

<span id="page-6-0"></span>Figure 2: Expert utilization throughout training for MoE-M (left) and MoE-L (right), comparing LBL, our method (SimBal), and a baseline with no load balancing. We measure the number of unique experts activated on our full 77M-token validation set over time. Without any balancing, the expert routing collapses to a smaller set of experts. Both LBL and SimBal maintain full expert avoid expert collapse. The no-loss baseline was truncated early.

<span id="page-6-2"></span>![](_page_6_Figure_3.jpeg)

![](_page_6_Figure_4.jpeg)

- (a) Redundancy Per Layer, Lower = Better (b) ∆ Redundant Expert Knowledge
- <span id="page-6-1"></span>

Figure 3: Analysis of expert redundancy in MoE-L models. (a) PES across different layers, our approach (blue) maintains significantly lower redundancy than LBL (orange). Darker = later in training. (b) Rate of change of PES during training, averaged over all layers. Redundancy occurs when many distinct experts see similar tokens, and is most likely to happen early in training, as we observe. We note that this is ą 0 at most points for LBL, suggesting it exacerbates redundancy during the majority of training.

trained with LBL have a mean perplexity of 14.051 with a standard deviation of 0.026. In comparison, SIMBAL achieves a mean perplexity of 13.691 with standard deviation 0.039. The mean SIMBAL performance is over 13 standard deviations lower than the perplexity of LBL, showing that our results are very statistically significant.

Finally, we examine sensitivity to the auxiliary loss coefficient (0.01, 0.1, 1.0), with results in Table [7.](#page-8-1) Based on our 5-seed runs on MoE-M, the effect is negligible, and we do not recommend tuning this hyperparameter.

### 4.3 Redundancy and Specialization in Experts

Motivated by [Dai et al.](#page-10-3) [\[2024\]](#page-10-3), we study expert specialization and redundancy. As described in Section [3.3,](#page-4-2) we measure these properties with Pairwise Expert Similarity (PES), in contrast to their expert dropout approach. In Figure [4,](#page-8-2) we validate the correlation between PES and their method, reproducing their redundancy analysis. By their metric, SIMBAL shows lower redundancy, as validation perplexity rises more sharply when top experts are dropped. However, such dropout-based metrics lack granularity and are prohibitively expensive for large-scale evaluation. PES instead provides a lightweight, scalable measure of redundancy, enabling per-layer, per-checkpoint analysis across all experts in parallel.

<span id="page-7-0"></span>Table 4: Model setup and performance.

| Model        | Dense-M | MoE-M  | MoE-M  | Dense-L | MoE-L  | MoE-L  |
|--------------|---------|--------|--------|---------|--------|--------|
| Balancing    | –       | LBL    | SimBal | –       | LBL    | SimBal |
| Perplexity Ó | 19.468  | 14.086 | 13.685 | 10.047  | 8.517  | 8.304  |
| Min PES Ó    | –       | 0.0255 | 0.0044 | –       | 0.0241 | 0.0028 |

<span id="page-7-1"></span>Table 5: Comparison of LBL-L and SimBal-L performance across benchmarks.

| Benchmark                           | LBL-L ˘ stderr | SimBal-L ˘ stderr |
|-------------------------------------|----------------|-------------------|
| ARC Challenge [Clark et al., 2018]  | 22.44% ˘ 1.22% | 23.21% ˘ 1.23%    |
| ARC Easy [Clark et al., 2018]       | 40.49% ˘ 1.01% | 41.16% ˘ 1.01%    |
| HellaSwag [Zellers et al., 2019]    | 35.45% ˘ 0.48% | 35.74% ˘ 0.48%    |
| PIAQ [Bisk et al., 2019]            | 66.49% ˘ 1.10% | 66.81% ˘ 1.10%    |
| WinoGrande [Sakaguchi et al., 2019] | 49.72% ˘ 1.41% | 52.49% ˘ 1.40%    |
| GLUE [Wang et al., 2018]            | 45.10% ˘ 1.98% | 51.73% ˘ 1.97%    |
| mean                                | 43.28%         | 45.19%            |

We hypothesize that SIMBAL produces less redundant experts than LBL. LBL enforces uniform distributions, leading to instability in early training as changing embeddings cause frequent routing shifts. Under near-uniform assignment, small input perturbations can reassign tokens, creating redundancy as many experts see similar tokens. We capture this effect by measuring changes in redundancy.

As shown in Figure [3\(b\),](#page-6-1) most redundancy in LBL (orange) arises early, coinciding with embedding volatility and unstable routing. Redundancy remains above zero through much of training, reinforcing that LBL amplifies it. In contrast, SIMBAL (blue) stabilizes quickly: while expert distributions adapt, they converge to consistently lower PES (Figure [3\(a\)\)](#page-6-2). Moreover, the rate of change remains near zero for most of training, showing that our method avoids the issues of LBL.

Final PES values are summarized in Table [4.](#page-7-0) To reduce sensitivity to outliers, we report the minimum PES across all layers, filtering out spikes in a single individual layer (common with LBL). We choose minimum, since we do not observe substantial dips in PES by layer, primarily jumps, and we wanted this metric to be as simple and intuitive as possible. SimBal consistently produces models with substantially lower minimum PES than LBL. Figure [5](#page-8-3) shows the rate of change in minimum PES over time.

#### <span id="page-7-2"></span>4.4 Inference-Time Expert Pruning

We further evaluate SIMBAL under inference-time *expert pruning*, following [Szatkowski et al.](#page-12-8) [\[2024\]](#page-12-8), where experts with assignment probabilities below a threshold are dropped at runtime. Results are presented in Table [4.4.](#page-7-2) SimBal produces less uniform assignments, allowing pruning to drastically improve efficiency with minimal perplexity cost. In contrast, LBL shows weaker synergy with pruning: while its performance drop is smaller (likely due to redundancy, similarly to Figure [4\(a\)\)](#page-8-4), improvements in throughput are limited. Notably, when experts below a weight of 0.15 are dropped (where both perplexities are most similar), SIMBAL achieves a 7.4% speedup (543s vs. 503s).

Table 6: Dynamic-K expert stage 3 selection [\[Szatkowski et al., 2024\]](#page-12-8) synergy with SIMBAL vs. LBL (perplexity and runtime on a full validation run, MoE-L). SIMBAL is able to preserve more performance when lower weighted experts are dropped for faster runtime.

| Dropped PpEq < | SimBal (PPL) | SimBal (s) | LBL (PPL) | LBL (s) |
|----------------|--------------|------------|-----------|---------|
| 0              | 8.317        | 620.927    | 8.536     | 619.657 |
| 0.1            | 8.364        | 571.147    | 8.542     | 575.121 |
| 0.15           | 8.598        | 503.065    | 8.621     | 543.027 |
| 0.2            | 9.380        | 472.915    | 9.057     | 495.200 |

<span id="page-8-4"></span>![](_page_8_Figure_0.jpeg)

![](_page_8_Figure_1.jpeg)

<span id="page-8-2"></span>Figure 4: Number of dropped top experts vs. validation loss, as proposed by [Dai et al.](#page-10-3) [\[2024\]](#page-10-3). SIMBAL exhibits lower redundancy, as shown by worse performance as more experts are dropped.

<span id="page-8-3"></span>Figure 5: Rate of change in minimum PES (over the layers of a model) over a training run, comparing LBL (higher perplexity) and SimBal (lower perplexity).

<span id="page-8-1"></span>Table 7: Performance across three scaling coefficients to SIMBAL. We find that the differences are not significant enough to warrant hyperparameter tuning.

| Model        | MoE-M  | MoE-M  | MoE-M  |
|--------------|--------|--------|--------|
| Coefficient  | 1.0    | 0.1    | 0.01   |
| Perplexity Ó | 13.716 | 13.685 | 13.687 |
| Min PES Ó    | 0.0045 | 0.0044 | 0.0050 |

