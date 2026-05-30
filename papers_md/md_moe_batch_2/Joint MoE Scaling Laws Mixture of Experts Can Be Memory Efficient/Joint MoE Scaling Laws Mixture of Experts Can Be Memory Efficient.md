# Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

Jan Ludziejewski \* 12 Maciej Pióro \* 23 Jakub Krajewski \* 12 Maciej Stefaniak <sup>1</sup> Michał Krutul <sup>12</sup> Jan Małaśnicki <sup>12</sup> Marek Cygan <sup>14</sup> Piotr Sankowski <sup>15</sup> Kamil Adamczewski <sup>26</sup> Piotr Miłoś <sup>27</sup> Sebastian Jaszczur <sup>12</sup>

#### **Abstract**

Mixture of Experts (MoE) architectures have significantly increased computational efficiency in both research and real-world applications of large-scale machine learning models. However, their scalability and efficiency under memory constraints remain relatively underexplored. In this work, we present joint scaling laws for dense and MoE models, incorporating key factors such as the number of active parameters, dataset size, and the number of experts. Our findings provide a principled framework for selecting the optimal MoE configuration under fixed memory and compute budgets. Surprisingly, we show that MoE models can be more memory-efficient than dense models, contradicting conventional wisdom. To derive and validate the theoretical predictions of our scaling laws, we conduct over 280 experiments with up to 2.7B active parameters and up to 5B total parameters. These results offer actionable insights for designing and deploying MoE models in practical large-scale training scenarios.

## 1. Introduction

Recently, language models have grown increasingly large, a trend accelerated by Mixture of Experts (MoE) techniques (Fedus et al., 2022; Du et al., 2022). MoE models are now widely adopted (Jiang et al., 2024; Dai et al., 2024) and are generally considered compute-efficient (Ludziejewski et al., 2024; Clark et al., 2022), though often considered memory-inefficient (Zadouri et al., 2023). However, the precise trade-offs between compute and memory efficiency have remained unclear so far.

Consider a motivating question: Can an MoE model be the optimal choice when constrained by a fixed memory budget, such as a single H100 node? Increasing the number of experts has a relatively minimal impact on the cost in FLOPs but can drastically increase memory requirements, often to prohibitive levels depending on the specific hardware and load.

In order to answer this question, we derive a *joint* scaling law for both dense and MoE models, accounting for key factors such as the number of active parameters, dataset size, and number of experts. This framework provides a rigorous analysis of model performance under strict memory constraints. Our findings reveal that, contrary to common assumptions, MoE models can be more memory-efficient than dense models—that is, MoE models with the same loss and training budget can have lower memory usage than dense models.

Our work is the first to provide detailed guidance on selecting the optimal number of experts for MoE models, balancing computational budget and memory. Our conclusions are based on extensive large-scale experiments with over 280 models, scaled up to 2.7B active parameters and up to 5B total parameters<sup>2</sup>. For a complete list of experiments, see Appendix E.

<sup>\*</sup>Core contributors <sup>1</sup>University of Warsaw <sup>2</sup>IDEAS NCBR <sup>3</sup>Institute of Fundamental Technological Research, Polish Academy of Sciences <sup>4</sup>Nomagic <sup>5</sup>MIM Solutions <sup>6</sup>Wroclaw University of Science and Technology <sup>7</sup>Institute of Mathematics, Polish Academy of Sciences. Correspondence to: Jan Ludziejewski <ludziej@mimuw.edu.pl>, Maciej Pióro <maciej.pioro@gmail.com>, Jakub Krajewski <gim.jakubk@gmail.com>, Sebastian Jaszczur <sebastian.jaszczur@gmail.com>.

<span id="page-0-0"></span><sup>&</sup>lt;sup>2</sup>We plan to open-source model checkpoints and the code.

In summary, the key contributions of this work are:

• We derive a joint scaling law for Mixture of Experts and dense models,

<span id="page-1-0"></span>
$$\mathcal{L}(N_{\text{act}}, D, \hat{E}) = a\hat{E}^{\delta} N_{\text{act}}^{\alpha + \gamma \ln(\hat{E})} + b\hat{E}^{\omega} D^{\beta + \zeta \ln(\hat{E})} + c, \tag{1}$$

where L represents the final training loss, Nact denotes the number of active parameters, D is the dataset size, Eˆ is a monotonic transformation of the number of experts (as defined in Equation [\(4\)](#page-2-0)), and c is the minimum achievable loss on the dataset, often called the irreducible entropy of the dataset.

- Based on the proposed scaling law, we show that the choice of the optimal number of experts (including dense models with E = 1) depends on specific computational and memory constraints, see Figure [1.](#page-1-0) Furthermore, we demonstrate how the optimal token-to-parameter ratio depends on E.
- We show that MoE can often be the preferred alternative to dense models, even if GPU memory is the constraining factor. We validate our theoretical findings by training a set of 1.1B-parameter models under identical compute and total memory budgets. The MoE models achieve a lower final loss, confirming their superior efficiency in practice. Moreover, we observe that MoE models not only have lower loss but also deliver higher performance during inference.

![](_page_1_Figure_7.jpeg)

Figure 1. (a) The loss of memory-constrained models predicted using our scaling law under a fixed training budget of 10<sup>22</sup> FLOPs. Each curve represents a different number of experts. The lines are truncated at compute-optimal points since undertrained models are both larger and worse in terms of loss, thus pointless in a memory-constrained scenario. Shaded areas indicate the memory-optimal number of experts for the corresponding memory budgets. (b) Experimental validation of the thesis that MoE can be memory-optimal. The marked area shows an interval in which training a compute-matched MoE achieves better loss than an overtrained dense model with the same number of total parameters (1.1B). Such an MoE is trained for longer and has fewer active parameters, making it more practical for inference.

## 2. Related Work

Mixture of Experts. Mixture of Experts (MoE) was introduced by [Jacobs et al.](#page-11-2) [\(1991\)](#page-11-2), who combined a gating network with a set of expert networks. [Shazeer et al.](#page-12-1) [\(2017\)](#page-12-1) applied MoE to an LSTM-based model [\(Hochreiter & Schmidhuber,](#page-11-3) [1997\)](#page-11-3), scaling the architecture up to 137 billion parameters. In Transformer-based LLMs, MoE is most often applied as a replacement for the feed-forward layer [\(Lepikhin et al.,](#page-11-4) [2020;](#page-11-4) [Shazeer et al.,](#page-12-2) [2018\)](#page-12-2). It replaces the feed-forward layer's MLP with a set of expert MLPs along with a router, which selects one or more MLPs for each token. With the recent surge in LLM research, MoE models are gaining even more traction. This is exemplified by the development of extremely large-scale models such as DeepSeek-R1 and Qwen2.5-Max [\(DeepSeek-AI et al.,](#page-10-4) [2025;](#page-10-4) [Team,](#page-12-3) [2024a\)](#page-12-3). In our work, we use the standard Switch MoE layer [\(Fedus et al.,](#page-10-0) [2022\)](#page-10-0), which routes each token to one expert and encourages even token-to-expert assignment via the addition of a differentiable load-balancing loss.

![](_page_2_Figure_1.jpeg)

Figure 2. (a) IsoFLOP profiles for selected training budgets, with compute-optimal points marked for each curve. (b) FLOP savings from switching from a compute-optimal dense model to a compute-optimal MoE. For instance, 40% savings at 1e20 FLOPs mean that an MoE matching the performance of a compute-optimal dense model trained with 1e20 FLOPs can be trained with just 6e19 FLOPs (60% of the dense's budget). The advantage of using MoE increases with larger models and expert counts.

Scaling Laws. Scaling laws refer to empirically derived equations that relate model loss to factors such as the number of parameters, the quantity of training data, or the computational budget. For dense Transformers, scaling laws were initially explored by Hestness et al. (2017) and Kaplan et al. (2020), who identified power-law relationships between the final loss, model size, and dataset size. Hoffmann et al. (2022) expanded these by incorporating variable cosine cycle lengths and adjusting the functional form of the equation:

<span id="page-2-3"></span><span id="page-2-1"></span>
$$\mathcal{L}(N_{\text{act}}, D) = mN_{\text{act}}^{\mu} + nD^{\nu} + c. \tag{2}$$

Scaling laws have also been applied to other architectures and training setups. Henighan et al. (2020) examined autoregressive modeling across multiple modalities, while Ghorbani et al. (2021) focused on machine translation. Frantar et al. (2023) studied the effects of pruning on vision and language Transformers, determining optimal sparsity given a fixed compute budget.

Clark et al. (2022) investigated scaling in MoE models, varying model size and the number of experts on a fixed dataset. They concluded that routed models are more efficient only up to a certain size. Their formula took the form:

<span id="page-2-2"></span>
$$\mathcal{L}(N_{\rm act}, \hat{E}) = a\hat{E}^{\delta} N_{\rm act}^{\alpha + \gamma \ln(\hat{E})},\tag{3}$$

where  $\hat{E}$  is a monotonic transformation of the number of experts E, defined as:

<span id="page-2-0"></span>
$$\frac{1}{\hat{E}} = \frac{1}{E - 1 + \left(\frac{1}{E_{\text{start}}} - \frac{1}{E_{\text{max}}}\right)^{-1}} + \frac{1}{E_{\text{max}}}.$$
 (4)

These analyses have since been extended by Ludziejewski et al. (2024) and Dai et al. (2024), who considered variable dataset size as well as the granularity of experts. In our work, we keep the experts non-granular; however, we treat the number of experts and the number of training tokens as variables. Sardana et al. (2024) assumes a fixed joint inference and training budget. We make similar assumptions; however, we consider accelerator memory as a limiting factor and extend the analysis to MoE models, which can serve as a more compute-friendly alternative to dense models. Yun et al. (2024) have focused on MoE inference optimality and measuring real hardware efficiency.

Concurrently to our work, Abnar et al. (2025) derived scaling laws for optimal sparsity while considering the interplay between training FLOPs and model size. They also investigated the relationship between pretraining loss and downstream performance, noting differences between MoE and dense models on certain tasks. In contrast, we analyze not only training FLOPs and model size but also inference cost and total memory usage. Additionally, we derive and utilize a principled method for scaling the learning rate with the number of experts and model size, along with describing further adjustments to enable researchers to use scaling laws economically and reliably.

## 3. Joint MoE Scaling Laws

We now derive the functional form of our joint scaling laws for both dense Transformers and MoE, relating the number of active model parameters Nact, training tokens D, and MoE experts E.

Fixed Number of Experts. Following [Hoffmann et al.](#page-11-7) [\(2022\)](#page-11-7) and established practices in the literature [\(Frantar et al.,](#page-10-5) [2023;](#page-10-5) [Kumar et al.,](#page-11-10) [2024;](#page-11-10) [Ludziejewski et al.,](#page-11-1) [2024\)](#page-11-1), we postulate the following form of the equation:

$$\mathcal{L}(N_{\text{act}}, D, E) = m(E)N_{\text{act}}^{\mu(E)} + n(E)D^{\nu(E)} + c(E), \tag{5}$$

assuming that if we fix the number of experts, the model's performance can be described using Equation [2.](#page-2-1) In the subsequent part, we will postulate how m, µ, n, ν, c depend on E, deriving the joint equation.

Constant Factor. c(E) represents irreducible loss caused by the inherent entropy of the dataset. Thus, it does not depend on the architecture (E in our case):

<span id="page-3-0"></span>
$$c(E) := c$$
.

Interaction of E with Model and Dataset Size. To quantify the interaction between the number of experts and other training parameters, we gather observations from related work:

- 1. Scaling the number of experts (E) can be described as a power law [\(Clark et al.,](#page-10-3) [2022\)](#page-10-3).
- 2. For a fixed number of training tokens (D), as model size (Nact) increases, the benefit of using an MoE diminishes [\(Clark](#page-10-3) [et al.,](#page-10-3) [2022\)](#page-10-3).
- 3. For a fixed model size (Nact), as the number of training tokens increases, the benefit of an MoE grows [\(Ludziejewski](#page-11-1) [et al.,](#page-11-1) [2024\)](#page-11-1).

Motivated by Observation 1, we set

$$m(E) = aE^{\delta}, \quad n(E) = bE^{\omega},$$

reflecting the power-law relation between E and the loss.

Additionally, to ensure flexibility in modeling Observations 2 and 3, we introduce an interaction with the exponents over Nact and D:

$$\mu(E) = \alpha + \gamma \ln(E),$$
  
$$\nu(E) = \beta + \zeta \ln(E).$$

Note that if we ignore the second and third terms in Equation [5,](#page-3-0) this yields a functional form identical to Equation [3.](#page-2-2)

Empirically, we observe a good fit for our formula, as described in Section [5.](#page-7-0) This shows that our proposed interactions between E, Nact, and D can accurately model the performance of MoE models.

Modeling of E. When the number of experts is small, a certain overhead—caused, for example, by interference from auxiliary losses—can overshadow the benefits of conditional computation. Additionally, employing a very large number of experts brings diminishing returns. To address these phenomena, we follow [Clark et al.](#page-10-3) [\(2022\)](#page-10-3) and use a transformation of the number of experts Eˆ as given in Equation [4.](#page-2-0)

Joint MoE Scaling Law. By combining these observations, we establish the final form of our scaling law:

<span id="page-3-1"></span>
$$\mathcal{L}(N_{\text{act}}, D, \hat{E}) = a\hat{E}^{\delta}N_{\text{act}}^{\alpha + \gamma \ln(\hat{E})} + b\hat{E}^{\omega}D^{\beta + \zeta \ln(\hat{E})} + c.$$
(6)

We fit the coefficients in Equation [6](#page-3-1) based on the results of our experiments; see Table [B.](#page-13-0) In Section [4,](#page-4-0) we present the outcomes and findings derived from the scaling laws. The details of the training runs, as well as the fitting procedure, are described in Section [5.](#page-7-0)

## <span id="page-4-0"></span>4. Compute and Memory Optimality

In this section, we employ our scaling laws to offer recommendations for optimal configurations in different training and inference scenarios. Refer to Appendix A for details on counting FLOPs, the relationships between active and total parameters, and other technical aspects.

#### 4.1. Compute Optimality

A model is considered compute-optimal if it achieves the lowest loss among models trained with the same compute budget F. To find such an optimal configuration, we optimize the following:

$$\arg \min_{N_{\rm act},D,E} \mathcal{L}(N_{\rm act},D,E)$$
 s.t.  $6N_{\rm act}D=F$ 

Finding 1. More experts → higher tokens-to-param ratio. Assume a fixed compute budget. In this scenario, when increasing the number of experts, it is optimal to decrease the number of active parameters and increase the number of training tokens

Optimal N and D Depend on the Number of Experts. Assuming a given number of experts E, the compute-optimal training configuration can be achieved by selecting the appropriate trade-off between training tokens and model size. IsoFLOP slices comparing the predicted loss with dataset size for selected compute budgets are plotted in Figure 2 (a).

accordingly (Table 1).

For any fixed E, our scaling law has the Chinchilla functional form of Equation 2. Thus, from Hoffmann et al. (2022), the compute-optimal number of tokens and active parameters for the budget F and the number of experts E are given by

$$N_{\rm act}^{\rm opt}(F) = G\left(\frac{F}{6}\right)^a, \quad D^{\rm opt}(F) = G^{-1}\left(\frac{F}{6}\right)^b,$$
 (7)

where

$$G = \left(\frac{\mu(E)m(E)}{\nu(E)n(E)}\right)^{\frac{1}{\mu(E) + \nu(E)}}, \quad a = \frac{\nu(E)}{\mu(E) + \nu(E)}, \quad b = \frac{\mu(E)}{\mu(E) + \nu(E)}.$$

We compare the optimal configurations for several compute budgets in Table 1.

Both from comparing the IsoFLOP slices (Figure 2) and the values listed in the table, we can see that the compute-optimal configuration for a given compute budget clearly depends on E, with MoE models requiring comparatively larger datasets and correspondingly fewer active parameters.

Mixture of Experts is Compute Optimal. We now compare the performance across various numbers of experts, with the respective values of tokens and active parameters optimized. As illustrated in Figure 2, we observe significant compute savings for MoE models compared to dense models, with a larger number of experts providing more pronounced benefits.

The higher efficiency of MoE in terms of training compute comes at the price of increased memory requirements. However, somewhat surprisingly, we find that MoE models can outperform dense models of the same size trained with the same amount of training compute—a result we describe in more detail in the next subsection.

<span id="page-4-1"></span>

| Training Budget | Experts | $N_{\rm act}^{\rm opt}$ | $D^{\mathrm{opt}}$ |
|-----------------|---------|-------------------------|--------------------|
| $10^{20}$       | 1       | 1.7B                    | 9.7B               |
|                 | 2       | 1.5B                    | 11.4B              |
|                 | 4       | 1.2B                    | 13.9B              |
|                 | 8       | 990M                    | 17 <b>B</b>        |
|                 | 16      | 810M                    | 20.7B              |
|                 | 32      | 669 <b>M</b>            | 24.9B              |
| $10^{21}$       | 1       | 5.7B                    | 29.3B              |
|                 | 2       | 5B                      | 33B                |
|                 | 4       | 4.4B                    | 38B                |
|                 | 8       | 3.8B                    | 44.3B              |
|                 | 16      | 3.3B                    | 51.2B              |
|                 | 32      | 2.85B                   | 58.4B              |
| $10^{22}$       | 1       | 18.8B                   | 88.6B              |
|                 | 2       | 17.4B                   | 96 <b>B</b>        |
|                 | 4       | 15.8B                   | 105.4B             |
|                 | 8       | 14.4B                   | 115.8B             |
|                 | 16      | 13.2B                   | 126.5B             |
|                 | 32      | 12.2B                   | 136.9B             |

Table 1. Example compute-optimal training configurations for MoE models. For every training budget, as the number of experts increases, the optimal  $D^{\mathrm{opt}}$  also increases while  $N_{\mathrm{act}}^{\mathrm{opt}}$  decreases.

#### 4.2. Model Memory Optimality

Often, it is insufficient to consider models solely from the perspective of compute optimality, as a compute-optimal model can be impractically large, preventing its deployment on available hardware. Additionally, it may only be possible to run a large model with a small batch size due to limited GPU memory, leading to low hardware utilization [\(He,](#page-11-11) [2022\)](#page-11-11). Therefore, it is natural to consider

### Finding 2. More experts → better performance.

For a given compute budget, increasing the number of experts always improves performance, provided the size of the model and the number of training tokens are adjusted (Figure [2](#page-2-3) (a)).

a straightforward extension to the notion of compute optimality, specifically model memory optimality. A model is said to be memory optimal if, among models trained with the same compute budget F and having at most M parameters, it achieves the lowest loss:

$$\operatorname*{arg\,min}_{N_{\mathrm{act}},D,E} \mathcal{L}(N_{\mathrm{act}},D,E)$$
 s.t.  $6N_{\mathrm{act}}D=F, \quad N_{\mathrm{total}} \leq M$ 

Note that model memory-matched dense and MoE models differ in the number of active parameters—MoE uses just a fraction of them. Intuitively, it should thus have worse performance. However, given some budget, it can be trained on more tokens, lowering the loss. Our scaling laws suggest that MoE models can be model memory optimal. We validate this claim by training a 1.1B dense model and a model size and FLOP matched E = {2, 4} counterparts (Figure [1\)](#page-1-0). Significantly, the MoE models attain lower loss even if the dense model is overtrained (i.e., after passing its compute-optimal token count).

#### 4.3. Total Memory Optimality

During autoregressive generation, a decoder-only model processes a single token while storing activations (keys and values) for previous tokens in the KV cache. In the case of multi-head attention, its size equals 2T × Nblocks × dmodel, where T is the number of tokens in the cache (possibly within multiple sequences in the batch). Including the cache size yields the optimization criterion:

### Finding 3. MoE can also be *memory* optimal.

A total-parameter-matched MoE model can outperform a dense model trained with the same compute budget (Figure [1\)](#page-1-0). Moreover, such an MoE model is more compute *and* memory efficient at inference.

$$\begin{split} & \underset{N_{\text{act}},D,E}{\min} \, \mathcal{L}(N_{\text{act}},D,E) \\ \text{s.t. } 6N_{\text{act}}D = F, \quad N_{\text{total}} + 2TN_{\text{blocks}}d_{\text{model}} \leq M \end{split}$$

For practical values of T, a fair comparison of memory requirements should include the size of the KV cache in addition to the model size. Figure [3](#page-6-0) (b) presents the optimal models for a given compute and varying memory constraints when the size of the KV cache is included. Importantly, MoE models compare more favorably to dense models in this graph, and as T increases, they outperform dense models at increasingly smaller model sizes. In Figure [1](#page-1-0) (b), the E = {2, 4} models employ a smaller KV cache, which means that if memory is constrained, the MoE model can store longer contexts or work with a larger batch size than the dense model.

#### 4.4. Inference Optimality

Large models, while capable, may also be too costly to operate due to their high computational demands. To account for this drawback, we can further assume that a model will process a number of tokens, Dinf, over its lifetime and find the best model whose demands do not exceed a predefined joint training and inference budget:

$$\begin{aligned} & \mathop{\arg\min}_{N_{\text{act}},D,E} \mathcal{L}(N_{\text{act}},D,E) \\ & \text{s.t. } 6N_{\text{act}}D + 2N_{\text{act}}D_{\text{inf}} = F. \end{aligned}$$

Figure [3](#page-6-0) (c) presents the optimal models for a given compute and varying memory constraints if a joint budget needs to accommodate both training and inference demands. We find that, in this scenario, MoE models outperform dense models at smaller scales than in simple compute optimality due to reduced inference FLOPs. The E = 2 and E = 4 models shown in Figure [1](#page-1-0) use 36% and 61% less FLOPs per token, respectively, than their dense counterparts.

![](_page_6_Figure_1.jpeg)

Figure 3. Predicted loss for various numbers of experts at a FLOPs budget F = 5 × 10<sup>22</sup>. The x-axis represents the size of the model in terms of the number of parameters (a) or the total memory budget for both model parameters and KV cache for 8192 tokens (b, c). Shaded areas indicate the optimal number of experts for the corresponding parameter or memory budget. (c) In addition to the KV cache, the inference cost on 100B tokens is included in the FLOPs budget of F = 5 × 10<sup>22</sup> .

### 4.5. Summary

The concepts of inference optimality and total memory optimality can naturally be combined. Figure [3\(](#page-6-0)c) presents a comparison between different numbers of experts, where the KV cache is included in the model's memory requirements and the compute budget is shared between training and inference. Finally, Figure [4](#page-7-1) and Table [2](#page-6-1) investigate the optimal E for a sample of model sizes, while including the KV cache and considering the inference cost.

For practitioners, as a simplification of our analysis, we propose a general rule of thumb:

Rule of Thumb. For a fixed total parameter count, an MoE model with E ≤ 8 experts outperforms a compute-optimal dense model if trained on E times more tokens while maintaining the same memory footprint.

For instance, a compute-optimal 1.1B model trained for 8B tokens will have worse loss than either a 2-expert, 1.1B total parameters MoE model trained on 16B tokens or a 4-expert, 1.1B total parameters MoE model trained on 32B tokens. At the same time the MoE models will require fewer FLOPs per token during inference.

<span id="page-6-0"></span>

|      | 24GB | 80GB | 640GB |
|------|------|------|-------|
| 1021 | 16   | ≥ 32 | ≥ 32  |
| 1022 | 4    | 16   | ≥ 32  |
| 1023 | 1    | 8    | ≥ 32  |
| 1024 | 1    | 1    | 16    |

<span id="page-6-1"></span>Table 2. Optimal E for different training budgets and three typical memory constraints, corresponding to an RTX4090 GPU, an H100 GPU, and an 8xH100 GPU node. We assume 16k tokens in the KV cache and bfloat16 for storing model weights and activations.

Note that in the scenario described by the rule of thumb, compute-matched MoE will generally have less than E-times larger dataset and will still surpass dense model (as in Figure [1](#page-1-0) (b)), but we wanted to keep this rule simple and conservative. Furthermore, while the rule may plausibly apply with E > 8, we prefer to conservatively limit it to E ≤ 8 due to the uncertainty of predicting the loss of highly overtrained models (i.e., with a large token-to-parameter ratio). A detailed comparison can be found in Figure [6,](#page-14-0) illustrating a stronger result where memory- and compute-matched MoE outperform compute-optimal dense models across scales.

It is important to recognize that such scaling depends on access to large datasets—a concern frequently raised in the context of scaling LLMs. While many leading organizations have demonstrated that data limitations can be overcome, the availability of large-scale datasets varies by organization and domain, particularly outside of NLP. Whether NLP datasets are effectively unlimited remains an open question beyond the scope of this work.

![](_page_7_Figure_1.jpeg)

<span id="page-7-1"></span>Figure 4. Investigation of the optimal number of experts for three different model sizes: 2B, 5B, and 10B; and in three different scenarios from left to right: simply measuring the model size, including the size of a KV-cache with 32k tokens, and including the inference cost of processing 100B tokens. Note that in the second graph, the memory constraint corresponds to the memory requirements of dense models with sizes 2B, 5B, and 10B, including the KV cache, while utilizing bfloat16 for both parameters and activations.

## <span id="page-7-0"></span>5. Fitting the Scaling Law

In this section, we present the details of the experiments and the procedure for fitting the scaling law parameters, see Table [B](#page-13-0) in the Appendix. These results are based on extensive large-scale empirical evidence, including over 280 models with up to 5B parameters, trained on a variety of compute budgets. For a comprehensive list of experiments, see Appendix [E.](#page-16-0)

#### 5.1. Model Hyperparameters

The selection of hyperparameters and training details is crucial for ensuring the robustness of scaling laws [\(Porian et al.,](#page-12-6) [2025;](#page-12-6) [Pearce & Song,](#page-11-12) [2024\)](#page-11-12). In our work, we employ a set of best practices and modern design choices, aiming to provide accurate predictions applicable to real-life practice.

All models used in this study are decoder-only Transformers trained on the highly filtered FineWeb-Edu [\(Penedo et al.,](#page-12-7) [2024\)](#page-12-7). We use a Transformer model with Switch [\(Fedus et al.,](#page-10-0) [2022\)](#page-10-0) layers, using standard values of router z-loss of 0.001 and load balancing loss of 0.01. The GPT-2 tokenizer [\(Radford et al.,](#page-12-8) [2018\)](#page-12-8) is employed. For better stability, weight initialization follows a truncated normal distribution with a reduced scale of 0.1, as suggested by [\(Fedus et al.,](#page-10-0) [2022\)](#page-10-0). Mixed precision training is used, with the attention mechanism, RoPE position embeddings [\(Su et al.,](#page-12-9) [2023\)](#page-12-9) and router always maintained at high precision. The models use the SwiGLU activation [\(Shazeer,](#page-12-10) [2020\)](#page-12-10) with hidden size equal to 3dmodel and activate one expert per token (unless the token is dropped due to limited capacity). For evaluation, we increase the capacity factor to ensure dropless processing of the tokens.

### 5.1.1. BATCH SIZE RAMP-UP

Performance of a deep learning optimization procedure can suffer as a result of using an exceedingly large batch size [\(Mc-](#page-11-13)[Candlish et al.,](#page-11-13) [2018\)](#page-11-13). To mitigate this potential issue, especially early in the training, we employ batch-size ramp-up. Similar strategies are used in contemporary LLM training runs [\(Rae et al.,](#page-12-11) [2022;](#page-12-11) [Dubey et al.,](#page-10-7) [2024\)](#page-10-7). We increase the batch size from 64K to 128K after 0.5B training tokens and further to 256K after 1B training tokens. Instead of utilizing noise scale as a critical batch size predictor [\(McCandlish et al.,](#page-11-13) [2018\)](#page-11-13) we opted for a straightforward grid to directly predict a transition point beyond which increasing batch size does not impair performance.

#### 5.1.2. LEARNING RATE SCALING

Kaplan et al. (2020) have shown that scaling laws for hyperparameters can be used to adjust them according to the size of the model in the case of dense Transformers. For MoE models, we find the literature inconclusive—while some (Dai et al., 2024) pretrain MoEs with a lower learning rate than corresponding dense models, others (Zoph et al., 2022) report better performance when fine-tuning MoEs with higher learning rates. To address this discrepancy, we derive a scaling law for the peak learning rate for MoE based on the number of active non-embedding parameters  $N_{act \setminus e}$  and the number of experts E:

$$LR(N_{act \setminus e}, E) = \exp(8.39 - 0.81 \ln(N_{act \setminus e}) - 0.25 \ln(E)),$$
 (8)

and use this equation to set the learning rate in our main scaling laws experiments. We fit the coefficients of the equation using the least squares method, minimizing the error between the prediction and the optimal learning rate from the experiment grid. In contrast to Kaplan et al. (2020), we use a linear transformation of the parameter count to predict the logarithm of the learning rate, instead

Finding 4. More experts → lower learning rate. Increasing the number of experts in MoE model should be accompanied by lowering the learning rate accordingly (Figure 7 in the Appendix).

of directly predicting the learning rate. This approach allows us to avoid the breakdown of the formula above  $10^{10}$  parameters, as mentioned in their work, where the predicted learning rate becomes negative. This phenomenon is independent of the actual fit and is simply a property of the formula used. Besides being well-defined in the extrapolation, we argue that optimal learning rates visibly follow this logarithmic trend, as seen in Figure 7 in the Appendix.

The second difference between our formula and the one by Kaplan et al. (2020) is the incorporation of the number of experts, allowing us to model the optimal behavior of this hyperparameter across dense models and different MoEs. This is an important detail that allows unbiased comparison among different models and ensures each one is optimally tuned. Furthermore, it allows us to answer the question of whether MoE should be trained with a lower or higher learning rate. While our formula accommodates both scenarios, we can clearly see in Figure 7 in the Appendix that increasing E requires lower learning rates, resulting in a negative value for the coefficient. Moreover, we verify this thesis by tuning the fit on E=1 and E=8, and validating it on interpolation at E=4 and extrapolation at E=32. In both instances, the validation predicts the optimal learning rate for the model configuration or a value with nearly the same performance.

In Figure 8 in the Appendix, we perform an ablation of this additional power law on E by repeating our entire fitting procedure without the E component. This shows, especially with extrapolation on E=32, that dependence on E is crucial, and its omission can impair the performance of MoEs. Further details about our scaling rule for learning rates can be found in the plots in Appendix D.

### 5.1.3. LEARNING RATE SCHEDULE

Hägele et al. (2024) suggests that a trapezoidal learning rate schedule can yield similar performance to other established methods, such as the cosine schedule. Additionally, it provides a valuable advantage when varying training duration, as intermediate checkpoints can be reused. With a cosine schedule, intermediate checkpoints introduce bias into the fit, according to the analysis of Kaplan et al. (2020) by Hoffmann et al. (2022). We employ a constant learning rate schedule with a linear warmup over the initial 130M tokens and with a linear decay from the peak learning rate to 0 over the final 20% of tokens. For each model size, longer runs reuse intermediate checkpoints from the shorter ones.

#### 5.2. Optimization of Formula Coefficients

Following Hoffmann et al. (2022), we use the LBFGS algorithm to optimize the coefficients of Equation 6. See Appendix B for details. We observe a good fit with  ${\rm RMSE}_v=0.0039$  on a held-out set of our 30 runs with the lowest loss, and  ${\rm RMSE}_t=0.0062$  on the training dataset. To further verify our formula, we train separate Chinchilla scaling laws (Equation 2) for different E using the same hyperparameters and the corresponding subset of the initializations grid. This approach serves as a lower bound for the loss of our joint formula on the training dataset, as it can emulate its coefficients; however, it is more prone to overfitting because effectively more parameters are utilized. Using this approach, we obtain a lower error on the training dataset of  ${\rm RMSE}_v^{\rm sep}=0.0059$  and marginally higher on the validation  ${\rm RMSE}_v^{\rm sep}=0.0041$ . We believe this is a strong confirmation that our joint formula is actually describing how variable E influences training.

In Figure 5, we visually verify the extrapolation of the joint fit. Prediction errors are categorized by different numbers of experts, highlighting that our joint formula is not biased for any specific E.

![](_page_9_Figure_1.jpeg)

<span id="page-9-0"></span>Figure 5. (a) Quality of the fit. The maximum absolute error on the held-out extrapolation set is 0.018. (b) Predicted loss compared to observed loss for E=1. (c) Predicted loss (dashed line) compared to observed loss for E=4. We can see that on the training dataset, the error increases in an undertrained setting (D/N < 1 — more tokens than parameters). However, this scenario is never practical from our perspective.

#### 6. Limitations and Future Work

In our work, we focus on the standard MoE variant, where the size of each expert matches the size of the feed-forward layer in the corresponding dense model. Some recent findings (Dai et al., 2024; Ludziejewski et al., 2024; Muennighoff et al., 2024; Team, 2024b) suggest that fine-grained MoE models are more efficient and may likely enhance the benefits we report for using MoE. Similarly, adopting a dropless MoE (Gale et al., 2022) approach, instead of relying on a capacity factor, could lead to further improvements. We leave the integration of these MoE improvements for future work.

Moreover, our Chinchilla-based optimality analysis utilizes FLOPs, which may not accurately reflect the wall-clock training time of models with different architectures. Although comparing models based on the total number of parameters, rather than active parameters, partially alleviates this issue due to the same memory bottleneck, different implementations and distributed training algorithms are not considered in this work.

We assumed the Chinchilla scaling law (Equation 2) as the basis for our formulas. While this is well-grounded in the literature, this formula is known to have limitations, particularly for extreme token-to-parameter ratios. Similarly, we observed a regression in the goodness of fit for some heavily undertrained or overtrained runs.

### 7. Conclusions

In this work, we derived the joint scaling laws for Mixture of Experts, relating the loss of the model to the number of parameters, the number of training tokens, and the number of experts. By considering both compute and memory constraints, as well as the expected inference workload, we demonstrated that MoE models can outperform dense models even when constrained by memory usage or total parameters, contrary to common assumptions and intuitions that MoE models are more memory-intensive than dense models.

Our analysis reveals how optimal training strategies shift as the number of experts varies. This provides a principled framework for selecting MoE hyperparameters under given constraints, highlighting the trade-offs between memory and compute performance.

## Acknowledgments

We would like to express sincere gratitude to Szymon Antoniak and Piotr Padlewski for their detailed comments and invaluable discussions. We also thank Konrad Staniszewski for his feedback on the draft of this paper.

We gratefully acknowledge the Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2024/017060. This research was partially supported by the ERC PoC Grant EXALT no. 101082299, the National Science Centre (NCN) Grant no. 2020/37/B/ST6/04179, the National Science Centre (NCN) Preludium Grant no. 2022/45/N/ST6/02222, the "European Lighthouse of AI for Sustainability" - ELIAS grant no. 101120237, and the NCBiR grant POIR.01.01.01-00-0433/20. Part of the experiments utilized computational resources provided by [Writer.](https://writer.com/)

## References

- <span id="page-10-6"></span>Abnar, S., Shah, H., Busbridge, D., Ali, A. M. E., Susskind, J., and Thilak, V. Parameters vs flops: Scaling laws for optimal sparsity for mixture-of-experts language models, 2025. URL <https://arxiv.org/abs/2501.12370>.
- <span id="page-10-3"></span>Clark, A., de las Casas, D., Guy, A., Mensch, A., Paganini, M., Hoffmann, J., Damoc, B., Hechtman, B., Cai, T., Borgeaud, S., van den Driessche, G., Rutherford, E., Hennigan, T., Johnson, M., Millican, K., Cassirer, A., Jones, C., Buchatskaya, E., Budden, D., Sifre, L., Osindero, S., Vinyals, O., Rae, J., Elsen, E., Kavukcuoglu, K., and Simonyan, K. Unified scaling laws for routed language models, 2022.
- <span id="page-10-2"></span>Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., Xie, Z., Li, Y. K., Huang, P., Luo, F., Ruan, C., Sui, Z., and Liang, W. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models, 2024.
- <span id="page-10-4"></span>DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL <https://arxiv.org/abs/2501.12948>.
- <span id="page-10-1"></span>Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., Zoph, B., Fedus, L., Bosma, M., Zhou, Z., Wang, T., Wang, Y. E., Webster, K., Pellat, M., Robinson, K., Meier-Hellstern, K., Duke, T., Dixon, L., Zhang, K., Le, Q. V., Wu, Y., Chen, Z., and Cui, C. Glam: Efficient scaling of language models with mixture-of-experts, 2022.
- <span id="page-10-7"></span>Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- <span id="page-10-0"></span>Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity, 2022.
- <span id="page-10-5"></span>Frantar, E., Riquelme, C., Houlsby, N., Alistarh, D., and Evci, U. Scaling laws for sparsely-connected foundation models, 2023.

- <span id="page-11-16"></span>Gale, T., Narayanan, D., Young, C., and Zaharia, M. Megablocks: Efficient sparse training with mixture-of-experts, 2022. URL <https://arxiv.org/abs/2211.15841>.
- <span id="page-11-9"></span>Ghorbani, B., Firat, O., Freitag, M., Bapna, A., Krikun, M., Garcia, X., Chelba, C., and Cherry, C. Scaling laws for neural machine translation, 2021.
- <span id="page-11-11"></span>He, H. Making deep learning go brrrr from first principles. 2022. URL [https://horace.io/brrr\\_intro.html](https://horace.io/brrr_intro.html).
- <span id="page-11-8"></span>Henighan, T., Kaplan, J., Katz, M., Chen, M., Hesse, C., Jackson, J., Jun, H., Brown, T. B., Dhariwal, P., Gray, S., Hallacy, C., Mann, B., Radford, A., Ramesh, A., Ryder, N., Ziegler, D. M., Schulman, J., Amodei, D., and McCandlish, S. Scaling laws for autoregressive generative modeling, 2020.
- <span id="page-11-5"></span>Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M. M. A., Yang, Y., and Zhou, Y. Deep learning scaling is predictable, empirically, 2017.
- <span id="page-11-3"></span>Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
- <span id="page-11-7"></span>Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. Training compute-optimal large language models, 2022.
- <span id="page-11-14"></span>Hägele, A., Bakouch, E., Kosson, A., Allal, L. B., Werra, L. V., and Jaggi, M. Scaling laws and compute-optimal training beyond fixed training durations, 2024. URL <https://arxiv.org/abs/2405.18392>.
- <span id="page-11-2"></span>Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. Adaptive mixtures of local experts. Neural Computation, 3(1): 79–87, 1991. doi: 10.1162/neco.1991.3.1.79.
- <span id="page-11-0"></span>Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., Hanna, E. B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L. R., Saulnier, L., Lachaux, M.-A., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T. L., Gervet, T., Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. Mixtral of experts, 2024.
- <span id="page-11-6"></span>Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models, 2020.
- <span id="page-11-10"></span>Kumar, T., Ankner, Z., Spector, B. F., Bordelon, B., Muennighoff, N., Paul, M., Pehlevan, C., Ré, C., and Raghunathan, A. Scaling laws for precision, 2024. URL <https://arxiv.org/abs/2411.04330>.
- Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.
- <span id="page-11-4"></span>Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. Gshard: Scaling giant models with conditional computation and automatic sharding, 2020.
- <span id="page-11-1"></span>Ludziejewski, J., Krajewski, J., Adamczewski, K., Pióro, M., Krutul, M., Antoniak, S., Ciebiera, K., Król, K., Odrzygó´zd´z, T., Sankowski, P., Cygan, M., and Jaszczur, S. Scaling laws for fine-grained mixture of experts. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 33270–33288. PMLR, 21–27 Jul 2024. URL <https://proceedings.mlr.press/v235/ludziejewski24a.html>.
- <span id="page-11-13"></span>McCandlish, S., Kaplan, J., Amodei, D., and Team, O. D. An empirical model of large-batch training, 2018. URL <https://arxiv.org/abs/1812.06162>.
- <span id="page-11-15"></span>Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Morrison, J., Min, S., Shi, W., Walsh, P., Tafjord, O., Lambert, N., Gu, Y., Arora, S., Bhagia, A., Schwenk, D., Wadden, D., Wettig, A., Hui, B., Dettmers, T., Kiela, D., Farhadi, A., Smith, N. A., Koh, P. W., Singh, A., and Hajishirzi, H. Olmoe: Open mixture-of-experts language models, 2024. URL <https://arxiv.org/abs/2409.02060>.
- <span id="page-11-12"></span>Pearce, T. and Song, J. Reconciling kaplan and chinchilla scaling laws, 2024. URL [https://arxiv.org/abs/2406.](https://arxiv.org/abs/2406.12907) [12907](https://arxiv.org/abs/2406.12907).

- <span id="page-12-7"></span>Penedo, G., Kydlícek, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Werra, L. V., and Wolf, T. The fineweb datasets: ˇ Decanting the web for the finest text data at scale, 2024. URL <https://arxiv.org/abs/2406.17557>.
- <span id="page-12-6"></span>Porian, T., Wortsman, M., Jitsev, J., Schmidt, L., and Carmon, Y. Resolving discrepancies in compute-optimal scaling of language models, 2025. URL <https://arxiv.org/abs/2406.19146>.
- <span id="page-12-8"></span>Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pre-training. 2018.
- <span id="page-12-11"></span>Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., Rutherford, E., Hennigan, T., Menick, J., Cassirer, A., Powell, R., van den Driessche, G., Hendricks, L. A., Rauh, M., Huang, P.-S., Glaese, A., Welbl, J., Dathathri, S., Huang, S., Uesato, J., Mellor, J., Higgins, I., Creswell, A., McAleese, N., Wu, A., Elsen, E., Jayakumar, S., Buchatskaya, E., Budden, D., Sutherland, E., Simonyan, K., Paganini, M., Sifre, L., Martens, L., Li, X. L., Kuncoro, A., Nematzadeh, A., Gribovskaya, E., Donato, D., Lazaridou, A., Mensch, A., Lespiau, J.-B., Tsimpoukelli, M., Grigorev, N., Fritz, D., Sottiaux, T., Pajarskas, M., Pohlen, T., Gong, Z., Toyama, D., de Masson d'Autume, C., Li, Y., Terzi, T., Mikulik, V., Babuschkin, I., Clark, A., de Las Casas, D., Guy, A., Jones, C., Bradbury, J., Johnson, M., Hechtman, B., Weidinger, L., Gabriel, I., Isaac, W., Lockhart, E., Osindero, S., Rimell, L., Dyer, C., Vinyals, O., Ayoub, K., Stanway, J., Bennett, L., Hassabis, D., Kavukcuoglu, K., and Irving, G. Scaling language models: Methods, analysis and insights from training gopher, 2022.
- <span id="page-12-4"></span>Sardana, N., Portes, J., Doubov, S., and Frankle, J. Beyond chinchilla-optimal: Accounting for inference in language model scaling laws, 2024. URL <https://arxiv.org/abs/2401.00448>.
- <span id="page-12-10"></span>Shazeer, N. Glu variants improve transformer, 2020. URL <https://arxiv.org/abs/2002.05202>.
- <span id="page-12-1"></span>Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017.
- <span id="page-12-2"></span>Shazeer, N., Cheng, Y., Parmar, N., Tran, D., Vaswani, A., Koanantakool, P., Hawkins, P., Lee, H., Hong, M., Young, C., Sepassi, R., and Hechtman, B. Mesh-tensorflow: Deep learning for supercomputers, 2018.
- <span id="page-12-9"></span>Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding, 2023. URL <https://arxiv.org/abs/2104.09864>.
- <span id="page-12-3"></span>Team, Q. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a.
- <span id="page-12-13"></span>Team, Q. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters", February 2024b. URL <https://qwenlm.github.io/blog/qwen-moe/>.
- <span id="page-12-5"></span>Yun, L., Zhuang, Y., Fu, Y., Xing, E. P., and Zhang, H. Toward inference-optimal mixture-of-expert large language models, 2024. URL <https://arxiv.org/abs/2404.02852>.
- <span id="page-12-0"></span>Zadouri, T., Üstün, A., Ahmadian, A., Ermi¸s, B., Locatelli, A., and Hooker, S. Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning. arXiv preprint arXiv:2309.05444, 2023.
- <span id="page-12-12"></span>Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., and Fedus, W. St-moe: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022.

## <span id="page-13-1"></span>A. Technical Details

### A.1. Counting Parameters

There are several ways to measure the size of a model. The two most important distinctions are whether total or active parameters are counted, and whether the parameters in the embedding and unembedding layers are included. Various papers assume different notations; notably, [Kaplan et al.](#page-11-6) [\(2020\)](#page-11-6) use nonembedding parameters, while [Hoffmann et al.](#page-11-7) [\(2022\)](#page-11-7) opt for the parameter count including embedding and unembedding. Throughout our work, we try to make it clear which way of counting we are using in each particular instance. When no additional information is given, Nact and Ntotal denote respectively active and total parameters, including the embedding and unembedding.

If we let dmodel be the hidden dimension of the model and dvocab be the vocabulary size (50, 257 in our case), then the following relations hold:

$$N_{\text{total}} = 2d_{\text{model}}d_{\text{vocab}} + (4 + 9E)N_{\text{blocks}}d_{\text{model}}^2$$
(9)

$$N_{\rm act} = 2d_{\rm model}d_{\rm vocab} + 13N_{\rm blocks}d_{\rm model}^2 \tag{10}$$

### A.2. Counting FLOPs

Based on [Sardana et al.](#page-12-4) [\(2024\)](#page-12-4), we assume the cost of training to be Ftraining = 6NactDtraining, and the cost of inference to be Finference = 2NactDinference. Due to the relatively small number (≤ 32) of experts used with implicit expert granularity of 1.0 [\(Ludziejewski et al.,](#page-11-1) [2024\)](#page-11-1), we can consider the memory and FLOPs cost of routing to be negligible, following [Clark et al.](#page-10-3) [\(2022\)](#page-10-3).

### A.3. Model Configs

The vast majority of our experiments use a simple rule for scaling the configuration, i.e., Nblocks = Nheads = dmodel/64 and assume these relations hold in all calculations. We base this rule on findings by [Kaplan et al.](#page-11-6) [\(2020\)](#page-11-6).

## <span id="page-13-0"></span>B. Fit Details

| a     | α       | δ       | γ      | b     | β       | ω      | ζ       | Estart | Emax     | c      |
|-------|---------|---------|--------|-------|---------|--------|---------|--------|----------|--------|
| 35.91 | −0.1889 | −0.2285 | 0.0098 | 35.98 | −0.1775 | 0.5529 | −0.0259 | 2.0732 | 290.4521 | 1.3637 |

Table 3. Fitted coefficients of our joined formula.

| E  | m       | µ       | n        | ν       | c      |
|----|---------|---------|----------|---------|--------|
| 1  | 30.3640 | −0.1817 | 53.9838  | −0.1965 | 1.3637 |
| 2  | 27.7982 | −0.1780 | 66.8401  | −0.2065 | 1.3637 |
| 4  | 24.8462 | −0.1731 | 87.7022  | −0.2192 | 1.3637 |
| 8  | 21.8330 | −0.1676 | 119.9126 | −0.2338 | 1.3637 |
| 16 | 19.0159 | −0.1617 | 167.5073 | −0.2494 | 1.3637 |
| 32 | 16.5424 | −0.1557 | 234.6726 | −0.2652 | 1.3637 |

Table 4. The fitted coefficients of our joint formula, Equation [6,](#page-3-1) reduced to the Chinchilla scaling law, Equation [2,](#page-2-1) for a given number of experts, E. We observe that the dataset exponent, ν, increases significantly. This is one of the reasons why compute-optimal parameter-to-token ratios change with E.

Following [Hoffmann et al.](#page-11-7) [\(2022\)](#page-11-7), we use the LBFGS algorithm with a learning rate of 1e−4 and a weight decay of 1e−5 to fit the coefficients of Equation [6,](#page-3-1) optimizing the Huber loss with δ = 0.01 over the set of our training runs described in the table in Appendix [E.](#page-16-0) Instead of removing outliers and underperforming models from the training set, we underweight them proportionally to the loss. Optimization hyperparameters were manually tuned to minimize error over the training dataset. The final fitted coefficients of Equation [6](#page-3-1) are within the boundaries of the grid of initializations given by: α ∈ {0.05, 0.25, 0.5}, β ∈ {0.05, 0.25, 0.5}, A ∈ {30, 100, 300}, B ∈ {30, 100, 300}, C ∈ {0.5, 1, 2},  $\delta \in \{-0.5, 0, 0.5\}, \gamma \in \{-0.5, 0, 0.5\}, \omega \in \{-0.5, 0, 0.5\}, \zeta \in \{-0.5, 0, 0.5\}$ . The selected coefficients were those with the lowest score, defined as the sum of RMSE on the training and a held-out extrapolation validation set. The formula in Equation 6 was calculated in logarithm, without any exponentials, using only linear transformations and the logsumexp operation. It was optimized to predict the logarithm of L, and parameters a, b, and c were optimized in logarithm. All these steps were taken to increase numerical stability and were essential for proper convergence.

### C. Compute- & Memory-Matched Models

![](_page_14_Figure_3.jpeg)

<span id="page-14-0"></span>Figure 6. Comparison between compute- and memory-matched models with different values of E. The corresponding total memory constraint for MoE models is derived from the compute-optimal model size for the dense model. Due to the nature of this constraint, we do not consider higher values of E, as their token-to-parameter ratio significantly exceeds the threshold within which we believe our scaling law applies. For instance, an MoE model with E=16 that matches a 1B dense model trained on 10B tokens in FLOPs and memory would have 155M activated parameters trained on 64B tokens. This results in a token-to-parameter ratio of approximately 414, surpassing the range covered by our dataset.

## <span id="page-15-2"></span>D. Learning Rate Scaling Fit

![](_page_15_Figure_2.jpeg)

<span id="page-15-0"></span>Figure 7. Visualization of the fit  $(E \in \{1, 8\})$  of our LR scaling rule, interpolation (E = 4) and extrapolation (E = 32).

![](_page_15_Figure_4.jpeg)

<span id="page-15-1"></span>Figure 8. Ablation for the LR scaling rule fit without considering the number of experts E. While performance on the training set  $(E \in \{1, 8\})$  looks acceptable, the extrapolation at E = 32 is clearly suboptimal, validating the need for considering E.

## <span id="page-16-0"></span>E. Experiments Listing

| Ntotal | Nattn_heads | Nblocks | dmodel | Nact | E  | D                                                               |
|--------|-------------|---------|--------|------|----|-----------------------------------------------------------------|
| 5.0B   | 16          | 16      | 1024   | 321M | 32 | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 3.8B   | 28          | 28      | 1792   | 1.3B | 4  | 11.1B, 5.6B, 2.8B, 2.0B                                         |
| 3.3B   | 11          | 21      | 1408   | 683M | 8  | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 3.0B   | 26          | 26      | 1664   | 1.1B | 4  | 80.0B, 64.0B, 48.0B, 32.0B, 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M |
| 2.7B   | 36          | 36      | 2304   | 2.7B | 1  | 9.2B, 5.5B, 2.8B, 2.0B, 1.4B, 980M                              |
| 2.6B   | 30          | 30      | 1920   | 1.6B | 2  | 5.4B, 2.7B                                                      |
| 2.6B   | 16          | 16      | 1024   | 321M | 16 | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 2.2B   | 28          | 28      | 1792   | 1.3B | 2  | 18.6B, 11.1B, 5.6B, 4.0B, 2.8B, 2.0B                            |
| 2.1B   | 12          | 12      | 768    | 169M | 32 | 8.0B, 4.0B, 2.0B, 1.0B, 500M                                    |
| 2.1B   | 10          | 16      | 1280   | 469M | 8  | 32.0B, 16.0B, 8.0B, 4.0B, 2.0B, 1.0B                            |
| 1.9B   | 22          | 22      | 1408   | 709M | 4  | 35.3B, 12.2B, 10.6B, 7.7B, 5.3B, 3.8B                           |
| 1.8B   | 11          | 21      | 1408   | 683M | 4  | 8.0B, 16.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 1.8B   | 26          | 26      | 1664   | 1.1B | 2  | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 1.6B   | 30          | 30      | 1920   | 1.6B | 1  | 5.4B, 2.7B                                                      |
| 1.4B   | 16          | 16      | 1024   | 321M | 8  | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 1.3B   | 28          | 28      | 1792   | 1.3B | 1  | 6.5B, 3.3B, 18.6B, 11.1B, 5.6B, 4.0B, 2.8B, 2.0B                |
| 1.3B   | 10          | 10      | 640    | 118M | 32 | 4.0B, 2.0B, 1.0B, 500M                                          |
| 1.2B   | 10          | 16      | 1280   | 469M | 4  | 32.0B, 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                      |
| 1.1B   | 12          | 12      | 768    | 169M | 16 | 8.0B, 4.0B, 2.0B, 1.0B, 500M                                    |
| 1.1B   | 26          | 26      | 1664   | 1.1B | 1  | 14.0B, 12.0B, 10.0B, 80.0B, 64.0B, 48.0B, 32.0B                 |
| 1.1B   | 26          | 26      | 1664   | 1.1B | 1  | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 1.1B   | 22          | 22      | 1408   | 709M | 2  | 3.8B, 49.8B, 24.9B, 12.5B, 6.2B, 3.1B, 1.6B, 778M               |
| 1.1B   | 22          | 22      | 1408   | 709M | 2  | 21.8B, 18.7B, 15.6B, 35.3B, 12.2B, 10.6B, 7.7B, 5.3B            |
| 1.1B   | 18          | 18      | 1152   | 426M | 4  | 31.0B, 25.9B, 20.7B, 10.4B, 5.2B, 2.6B, 1.3B                    |
| 1.1B   | 11          | 21      | 1408   | 683M | 2  | 32.0B, 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                      |
| 890M   | 24          | 24      | 1536   | 890M | 1  | 9.9B, 5.0B                                                      |
| 850M   | 20          | 20      | 1280   | 555M | 2  | 16.0B, 8.0B                                                     |
| 774M   | 16          | 16      | 1024   | 321M | 4  | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 709M   | 22          | 22      | 1408   | 709M | 1  | 35.3B, 12.2B, 10.6B, 7.7B, 5.3B, 3.8B, 12.5B, 6.2B              |
| 705M   | 10          | 16      | 1280   | 469M | 2  | 32.0B, 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                      |
| 683M   | 11          | 21      | 1408   | 683M | 1  | 32.0B, 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                      |
| 671M   | 10          | 10      | 640    | 118M | 16 | 4.0B, 2.0B, 1.0B, 500M                                          |
| 664M   | 8           | 8       | 512    | 79M  | 32 | 2.0B, 1.0B, 500M                                                |
| 615M   | 12          | 12      | 768    | 169M | 8  | 8.0B, 4.0B, 2.0B, 1.0B, 500M                                    |
| 555M   | 20          | 20      | 1280   | 555M | 1  | 16.0B, 8.0B                                                     |
| 472M   | 16          | 16      | 1024   | 321M | 2  | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 469M   | 10          | 16      | 1280   | 469M | 1  | 32.0B, 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                      |
| 376M   | 10          | 10      | 640    | 118M | 8  | 4.0B, 2.0B, 1.0B, 500M                                          |
| 362M   | 8           | 8       | 512    | 79M  | 16 | 2.0B, 1.0B, 500M                                                |
| 360M   | 12          | 12      | 768    | 169M | 4  | 8.0B, 4.0B, 2.0B, 1.0B, 500M                                    |
| 321M   | 16          | 16      | 1024   | 321M | 1  | 16.0B, 8.0B, 4.0B, 2.0B, 1.0B, 500M                             |
| 289M   | 11          | 11      | 704    | 142M | 4  | 4.5B, 2.3B, 1.1B                                                |
| 285M   | 9           | 9       | 576    | 97M  | 8  | 3.3B, 1.7B                                                      |
| 282M   | 13          | 13      | 832    | 201M | 2  | 6.4B, 3.2B, 1.6B, 800M                                          |
| 233M   | 12          | 12      | 768    | 169M | 2  | 8.0B, 4.0B, 2.0B, 1.0B, 500M                                    |
| 228M   | 10          | 10      | 640    | 118M | 4  | 4.0B, 2.0B, 1.0B, 500M                                          |
| 211M   | 8           | 8       | 512    | 79M  | 8  | 2.0B, 1.0B, 500M                                                |
| 169M   | 12          | 12      | 768    | 169M | 1  | 8.0B, 4.0B, 2.0B, 1.0B, 500M                                    |
| 154M   | 10          | 10      | 640    | 118M | 2  | 4.0B, 2.0B, 1.0B, 500M                                          |
| 135M   | 8           | 8       | 512    | 79M  | 4  | 2.0B, 1.0B, 500M                                                |
| 118M   | 10          | 10      | 640    | 118M | 1  | 4.0B, 2.0B, 1.0B, 500M                                          |
| 98M    | 8           | 8       | 512    | 79M  | 2  | 2.0B, 1.0B, 500M                                                |
|        | 8           | 8       | 512    | 79M  | 1  | 2.0B, 1.0B, 500M                                                |