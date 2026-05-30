# <span id="page-0-0"></span>Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

Zihan Wang1\*, Rui Pan2\*, Jiarui Yao<sup>2</sup> , Robert Csord ´ as´ 3 , Linjie Li<sup>4</sup> , Lu Yin<sup>5</sup> Jiajun Wu<sup>3</sup> , Tong Zhang<sup>2</sup> , Manling Li1† , Shiwei Liu6†

<sup>1</sup>Northwestern University <sup>2</sup>University of Illinois Urbana-Champaign <sup>3</sup>Stanford University <sup>4</sup>University of Washington <sup>5</sup>University of Surrey <sup>6</sup>University of Oxford

## Abstract

We propose Chain-of-Experts (CoE), a new Mixture-of-Experts (MoE) architecture that introduces sequential expert communication within each layer. Unlike traditional MoE models, where experts operate independently in parallel, CoE processes tokens iterative across a chain of experts inside a layer. To support dynamic expert selection across iterations, CoE employs a dedicated router at each iteration step within a layer. This design allows tokens to re-evaluate and select different experts during each iteration, rather than being statically assigned. As a result, CoE introduces a flexible routing mechanism that increases the diversity of expert combinations and enriches the model's representational capacity. CoE demonstrates improved performance under fixed compute: on Math reasoning tasks, it reduces validation loss from 1.20 to 1.12 compared to a standard MoE. Beyond performance, CoE offers a new *scaling axis*—depth through expert iteration—which complements conventional width/depth scaling. For example, using 2× iterations matches the performance of 3× expert selections (in width), while reducing memory usage by 17.6–42% relative to other scaling strategies. Our analysis reveals that CoE's benefits stem from its iterative residual structure and enhanced expert specialization empowered by iterative routing, which together unlock more expressive representations. Code is available at <https://github.com/ZihanWang314/coe>.

# 1 Introduction

Mixture-of-Experts (MoE) architectures have emerged as a dominant strategy for scaling large language models (LLMs), delivering significant gains in compute efficiency by sparsely activating a small subset of expert networks per input token [\[1,](#page-9-0) [2,](#page-9-1) [3,](#page-9-2) [4,](#page-9-3) [5\]](#page-9-4). This sparsity enables models to scale their parameter count independently from runtime cost, allowing for improved memory utilization and throughput during inference. Recent MoE systems have demonstrated strong empirical performance across a wide range of domains—including language [\[6,](#page-9-5) [7,](#page-10-0) [8,](#page-10-1) [9\]](#page-10-2), reasoning [\[10,](#page-10-3) [11\]](#page-10-4), and multimodal vision-language tasks [\[12,](#page-10-5) [13\]](#page-10-6).

Progress in MoE research has largely centered on scaling up expert capacity [\[14,](#page-10-7) [15\]](#page-10-8), improving routing algorithms [\[6,](#page-9-5) [16,](#page-10-9) [5\]](#page-9-4), and enhancing training stability [\[12\]](#page-10-5). These approaches often rely on a shared architectural assumption: *experts are conditionally and independently activated in parallel*, with no explicit interaction between them. While this design maximizes parallelism and system efficiency, it may also constrain the model's ability to exploit complementary reasoning patterns across

<sup>∗</sup>Equal contribution.

<sup>†</sup>Equal advising.

<span id="page-1-0"></span>![](_page_1_Picture_0.jpeg)

Figure 1: Comparison between Mixture-of-Experts (MoE) and Chain-of-Experts (CoE). Under the same model depth and parameters, CoE enables iterative expert communication to offer more flexible expert choice compared to MoE.

experts. As a result, existing MoE models could underutilize their available capacity, particularly for complex tasks that benefit from multi-expert coordination.

We challenge this independence assumption by introducing **Chain-of-Experts** (**CoE**) (Figure 1), a new MoE framework that enables *sequential communication* among intra-layer experts through iterative processing. While keeping the total number of experts used to process a token within a layer unchanged, CoE allows experts to process tokens in a "relay race" manner rather than independently: experts receive the intermediate representation from their predecessors, process it, and pass it to their successors, enabling richer expert composition and higher effective depth. Furthermore, sequential expert chains introduces a new scaling axis—*depth through iteration*—which could complement or even surpasses traditional scaling via width or layer depth.

In our experiments, under equivalent compute, CoE yields better performance (e.g., math validation loss drops from 1.20 to 1.12), higher efficiency (17.6–42% lower memory usage with comparable accuracy), and greater specialization (823× more effective expert combinations). These gains stem from a simple change: **iterative MoE layers with independent routing and residual connections at each iteration**. Our design is grounded in prior observations that experts often learn complementary roles and are highly specialized for different uses [17]. We empirically show that CoE consistently outperforms standard MoEs on reasoning-intensive tasks, particularly under compute-constrained regimes. We further analyze the impact of iteration count, gating design, and communication depth, offering a new lens on scaling modular language models efficiently.

### 2 Background: Mixture-of-Experts Transformers

We begin by formalizing the output computation of a standard Mixture-of-Experts (MoE) layer, a widely adopted mechanism to increase model capacity without a proportional increase in computation cost. In a typical Transformer architecture, certain Feed-Forward Network (FFN) sublayers are replaced by sparse MoE modules, where only a small subset of the available experts are activated per token.

Let  $x \in \mathbb{R}^d$  denote the input token embedding, and let  $\{E_i(\cdot)\}_{i=1}^N$  be a set of N expert networks, each structurally identical to a standard FFN. A gating function determines which  $K \ll N$  experts are selected for a given input, assigning a nonzero routing weight  $g_i$  to each selected expert. The output of the MoE layer is computed as:

$$y = \sum_{i=1}^{N} g_i \cdot E_i(x), \tag{1}$$

where the gating weights  $g_i$  are defined as:

$$g_i = \begin{cases} s_i, & s_i \in \text{TopK}(\{s_j\}_{j=1}^N, K), \\ 0, & \text{otherwise}, \end{cases} \quad \text{with} \quad s_i = \text{Softmax}(e_i^\top x). \tag{2}$$

Here,  $e_i \in \mathbb{R}^d$  is the router vector associated with the *i*-th expert. The dot product  $e_i^\top x$  measures the affinity between the input token and expert  $E_i$ . The TopK operator selects the K experts with the highest affinity scores, and the Softmax normalizes these scores over all experts. This gating

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2: **Illustration of Chain-of-Experts.** MoE Top-K experts operate in parallel without interaction, CoE enables the same number of experts to process sequentially via intermediate representations, allowing deeper network processing with the same per-token expert processing. Residual connections are enabled and iteration-independent routers are used for more effective training.

mechanism ensures sparsity: only K experts are active per token, significantly reducing compute and memory usage.

Some recent MoE variants [18] introduce *shared experts* across all tokens or layers to improve generalization and parameter efficiency. In such cases, the MoE output is extended as follows:

$$y = \sum_{i=1}^{M} \hat{E}_i(x) + \sum_{i=1}^{N} g_i \cdot E_i(x),$$
(3)

where  $\{\hat{E}_i(\cdot)\}_{i=1}^M$  denotes a set of shared experts that are applied uniformly to all tokens, while  $\{E_i(\cdot)\}_{i=1}^N$  are sparsely gated token-specific experts. Together, these mechanisms define the standard MoE formulation used in many scalable Transformer-based language models [1,3,4,6]. Our work builds on top of this foundation but departs from the conventional parallel and independent expert structure by introducing sequential communication across selected experts.

## 3 Chain-of-Experts: Communicative Expert Processing

While standard MoE routes each token through a fixed set of experts in a single forward pass, this design inhibits communication across experts within each layer. Intuitively, a token might benefit from passing through multiple experts in sequence—allowing each to refine the representation conditioned on what others have already processed.

To capture this notion of iterative processing, we propose an expert chaining mechanism that processes tokens over C communication steps, named Chain-of-Experts (CoE), as an alternative of current MoE layers. As shown in Figure 2, different from MoE layers, the token is re-routed to a new set of experts, each seeing the hidden state produced in the previous step. Formally, given an input token embedding x, we initialize the first hidden state as:

$$x^{(0)} = x, (4)$$

Then, at each iteration  $t = 1, \dots, C$ , we compute the next hidden state as:

$$x^{(t)} = \sum_{i=1}^{N} g_{t,i} \cdot E_i(x^{(t-1)}) + \mathbb{I}_r \cdot x^{(t-1)}, \tag{5}$$

where  $E_i$  is the *i*-th expert and  $g_{t,i}$  is the gating weight at step t. The term  $\mathbb{I}_r \cdot x^{(t-1)}$  denotes a residual connection, which we include by setting  $\mathbb{I}_r = 1$ . This simple addition helps preserve

information and stabilizes updates across iterations. The output after C iterations is:

$$y = x^{(C)}. (6)$$

This formulation naturally introduces expert communication—each expert in round t builds on what was computed in round t-1. Instead of forcing all experts to work independently, the model learns to decompose reasoning across steps. To ensure computational cost remains comparable to standard MoE, we could select only K/C experts at each iteration.

**Iteration-based Independent Routing.** To enable tokens to reevaluate and make adaptive routing decisions based on progressively refined hidden states, we assign a separate router to each iteration. Specifically, the gating weight  $g_{t,i}$  is defined as:

$$g_{t,i} = \begin{cases} s_{t,i}, & s_{t,i} \in \text{TopK}(s_{t,j} \mid 1 \le j \le N, K/C) \\ 0, & \text{otherwise} \end{cases}, \tag{7}$$

where  $s_{t,i} = \text{Softmax}(e_{t,i}^{\top}x^{(t-1)})$ . This independent routing mechanism across iterations allows each step to adapt freely: experts engaged in later iterations can refine, reinterpret, or build upon the intermediate representations produced earlier, enabling a form of progressive, layered processing.

Our iterative framework transforms MoE from a shallowly parallel system into a sequential expert reasoning process, while preserving its sparsity and modularity. When aligning both depth and total expert parameters, CoE offers a richer set of expert combinations compared to standard MoE by enabling expert reuse. As illustrated in Figure 1, we compare a two-layer Transformer equipped with MoE, where each layer contains four experts and activates two per token. With CoE, under the same parameter budget, we can achieve comparable depth using only a single Transformer layer with two communication steps. This setup allows more diverse expert compositions without significantly increasing computational cost. It is worth noting that Universal Transformers [19], such as MoEUT [15], also adopt a form of parameter reuse, while occurring across different Transformer layers, whereas CoE reuses experts within the same layer.

### 4 Experimental Setup

#### 4.1 Dataset and Metrics

We conduct our experiments on the general open-domain SlimPajama [20] and the reasoning domain-specific MetaMathQA [21] datasets. We train models on them separately and jointly (1:1 proportion) to evaluate the effect of sequential expert communication across domains. To assess the generalization performance, we track validation loss during training to measure optimization efficiency and convergence speed, and evaluate zero-shot performance on four standard benchmarks which span reasoning and commonsense tasks, including ARC-E [22], HellaSwag [23] and PIQA [24], with (normalized or soft) answer accuracy<sup>1</sup>.

#### 4.2 Training Settings

Training is performed using the AdamW optimizer with a learning rate of 3e-4, weight decay of 0.01, and betas set to (0.9, 0.95). We use a linear learning rate schedule with 10% warmup. Models are trained using a batch size of 64 and a sequence length of 512. To stabilize training, we apply gradient clipping with a threshold of 1.0.

#### 4.3 Model Configuration

Our model is built upon the **DeepSeek-V2-Lite** [25] architecture, following the same design while scaling down to a smaller configuration with 544 million parameters (excluding embeddings) to facilitate faster convergence and enable efficient experimental validation. The transformer backbone comprises 4 layers, each with hidden size 1024 and 8 attention heads. All layers are configured as MoE layers with a total of 63 routed experts and 1 shared expert. For each token, the router selects 8 routed experts and the shared expert, and each expert has an intermediate size of 704. In our CoE setup, we apply 2 iterations of expert processing with inner residual connections and enable independent gating per iteration. This configuration ensures a fair comparison between traditional MoE and CoE under similar parameter budgets and architectural depth.

<sup>&</sup>lt;sup>1</sup>https://github.com/EleutherAI/lm-evaluation-harness/issues/1396

<span id="page-4-1"></span>Table 1: General benchmark performance comparison between CoE and MoE. Under fixed expert compute across training sources, CoE (K=4, C=2) generally achieves better performance than MoE (K=8, C=1) on most settings, especially when trained and evaluated on mathematical reasoning datasets.

| Training Dataset  | Model        | ARC-E |       | HellaSwag |       | PIQA  |       |
|-------------------|--------------|-------|-------|-----------|-------|-------|-------|
|                   |              | Acc   | Norm  | Acc       | Norm  | Acc   | Norm  |
| SlimPajama        | MoE K=8, C=1 | 27.2% | 26.4% | 25.8%     | 25.1% | 52.9% | 51.0% |
|                   | CoE K=4, C=2 | 28.1% | 26.8% | 26.0%     | 25.1% | 53.1% | 51.1% |
| MetaMathQA        | MoE K=8, C=1 | 26.4% | 25.8% | 26.1%     | 26.0% | 54.6% | 52.3% |
|                   | CoE K=4, C=2 | 26.5% | 26.0% | 25.9%     | 26.3% | 54.5% | 51.4% |
| Combined Training | MoE K=8, C=1 | 26.4% | 28.0% | 26.5%     | 26.4% | 56.2% | 54.2% |
|                   | CoE K=4, C=2 | 27.7% | 28.2% | 26.5%     | 26.8% | 55.2% | 53.3% |

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

![](_page_4_Figure_3.jpeg)

- (a) Validation loss with same expert budget. (b) Iteration effect under sparse and dense setting.

Figure 3: CoE reduces validation loss more effectively than MoE under equal expert compute, specifically in sparse settings. CoE (K=4, C=2) outperforms MoE (K=8, C=1) with the same per-token expert processing (left). The benefit is amplified in sparse routing, where communication fosters specialization, but diminishes in dense settings where all experts are active (right).

#### 4.4 System and Infrastructure

All experiments are implemented within PyTorch, and we also borrow the Fully-Sharded Data Parallel (FSDP) Trainers from the veRL framework[2](#page-0-0) [\[26\]](#page-11-2), extending them to support multi-round expert execution and fine-grained token-level logging. The full detailed experimental code will be opensourced soon. We conduct training on NVIDIA H100 GPUs, and each run completes within one GPU hour, enabling reproducibility without the need for large-scale compute infrastructure.

## 5 Experimental Results and Analysis

In this section, we discuss several research questions that guide our investigation into how CoE may enable more effective language modeling:

- 1. Section [5.1:](#page-5-0) under the same compute budget, can the *communicative token processing* mechanism enhance language modeling compared to parallel processing?
- 2. Section [5.2:](#page-5-1) when scaling up computation, can expert iteration could serve as a better scaling factor compared to existing factors, such as model depth, width, expert choice count?
- 3. Section [5.3:](#page-6-0) what design choices could make sequential expert processing effective?
- 4. Section [5.4:](#page-7-0) does the expert communication mechanism enable step-wise expert specialization?
- 5. Section [5.5:](#page-8-0) does chain-of-expert processing exhibit a theoretical advantage in combinatorics and representational flexibility that explains its improved efficiency?

We explore these questions in the following subsections below. We also investigate the effect of shared experts in the Supplementary.

<sup>2</sup> <https://github.com/volcengine/verl>

<span id="page-5-2"></span>![](_page_5_Figure_0.jpeg)

Figure 4: Depth scaling comparison. CoE (C=2) with 4 layers matches the performance of deeper MoE models (L=8 or 12) with significantly lower memory and parameter usage.

<span id="page-5-3"></span>![](_page_5_Figure_2.jpeg)

Figure 5: Width scaling comparison. CoE (C=2) outperforms MoE variants with increased expert selection (K=16 or 24), while using similar resources.

#### <span id="page-5-0"></span>5.1 Communicative Processing can be Better than Parallel Processing in MoEs

We begin by discussing whether communicative processing could lead to better language modeling by setting a fixed expert-processing count at 8 per layer, adapting the communication steps C to explore whether models can benefit from expert communication. As shown in Figure [3a,](#page-4-0) CoE (C = 2, K = 4) consistently outperforms MoE (K = 8) with equal expert processing per token, reducing validation loss from 1.20 to 1.12 while exhibiting faster convergence. This advantage also generalizes beyond loss curves: as summarized in Table [1,](#page-4-1) CoE outperforms or achieves comparable performance as MoE on multiple downstream benchmarks across ARC-E, HellaSwag, PIQA. Due to the small model size, the comparison has not yet shown a significant gap, but it has already demonstrated the potential of CoE under different settings.

Interestingly, we find that CoE's advantage is primarily realized in sparse configurations. As shown in Figure [3b,](#page-4-0) when the model selects only a subset of experts per step (sparse routing), increasing the number of communication steps C leads to a noticeable improvement in validation loss. In contrast, under dense settings where all experts are always active (e.g., Total=K=8, C=2), iterative processing provides limited gain over one-shot routing. We hypothesize that this is because sparsity encourages expert specialization, allowing different iterations to focus on refining different token aspects. Without sparsity, the repeated processing simply deepens the computation path without introducing additional diversity, diminishing the benefit of communication. We further investigate this phenomenon in Section [5.4,](#page-7-0) where we show that different iteration lead to diverse expert sets.

#### <span id="page-5-1"></span>5.2 Expert Communication Steps can Serve as a Better Scaling Factor

To confirm the scaling effect of expert communication, we compare the communication step count (C) in CoE to conventional scaling dimensions in MoE architectures, such as network depth and expert selection width. Across three controlled experiments on the Math domain, we find that increasing communication steps in CoE can offer a more efficient scaling path, gaining comparable or better performance with lower memory and compute overhead.

(a) CoE can match deeper MoE with lower memory cost. To control for network depth, we fix the number of experts (N=8) and the number of experts selected per token (K=8), and scale the

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 6: Matched performance with fewer experts. CoE (N=48) achieves similar performance to MoE (N=64) while reducing memory usage by 17.6%.

number of transformer layers from 4 to 8 and 12. We compare these models to a 4-layer CoE model with C=2 expert communication steps. As shown in Figure [4,](#page-5-2) CoE achieves similar performance to MoE (L=12) while reducing memory usage by 42%. Unlike deeper MoE variants, CoE maintains constant parameter count and layer depth, reducing memory overhead with similar training time.

(b) CoE can outperform wider MoE under equal selection budget. We next scale the width of expert selection by varying the chosen experts each layer K from 8 to 16 and 24 in standard MoE, while keeping T=64 and L=4. In contrast, CoE retains K=8 but introduces communication (C=2). As shown in Figure [5,](#page-5-3) CoE achieves better convergence while consuming comparable memory and compute. This indicates that increasing C could be a more effective way to expand expert processing than simply increasing K.

(c) CoE delivers matched performance with fewer total experts. We further present that CoE can reduce the total number of experts while preserving performance. In Figure [6,](#page-6-1) CoE with C=2, K=4, and N=48 achieves similar validation loss to MoE with K=8 and N=64, but reduces memory usage by 17.6%. This demonstrates that CoE can achieve compute-efficient generalization by improving expert reuse instead of relying on scale alone.

Together, these results suggest that expert communication steps (C) in CoE offer a more efficient and scalable way to expand model capacity. Unlike traditional depth or width scaling which often increases memory footprint and compute cost, scaling C improves expert reuse and compositional depth without growing parameter count or memory usage.

### <span id="page-6-0"></span>5.3 Key Components Enabling Effective Sequential Expert Processing

To understand what makes sequential expert routing effective in CoE, we conduct an ablation study focusing on two key architectural choices: iteration-specific gating and inner residual connections. Both components are designed to improve expert compositionality and training stability. We compare ablated variants against standard CoE and a strong MoE baseline.

Iteration-specific gating enables per-step specialization. A central hypothesis of CoE is that each communication step should dynamically route tokens to different experts based on their evolving representation. To test this, we construct a variant that removes iteration-specific gating and instead reuses the same expert selection across all steps. Formally, the update rule becomes:

$$x^{(0)} = x,$$

$$x^{(t)} = \sum_{i=1}^{M} \hat{\mathbf{E}}_{i}(x^{(t-1)}) + \sum_{i=1}^{N} g_{i} \cdot \mathbf{E}_{i}(x^{(t-1)}) + \mathbb{I}_{r} \cdot x^{(t-1)}, \quad t = 1, \dots, C,$$

$$y = x^{(C)},$$
(8)

where the gating weights g<sup>i</sup> are fixed across iterations. This design removes the model's ability to condition expert routing on intermediate computation states.

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 7: Ablation study. Removing either iteration-specific gating or inner residuals significantly reduces performance. Both components are critical to the effectiveness of sequential expert routing in CoE.

As shown in Figure [7,](#page-7-1) this shared-gating variant significantly underperforms: its validation loss quickly plateaus around 1.5, worse than both standard CoE and the MoE baseline. These results highlight the importance of step-specific routing in CoE—without it, the model fails to leverage the compositional benefits of iterative reasoning.

Inner residuals stabilize multi-step refinement. We also investigate how residual connections should be applied across communication steps. In standard CoE, residuals are applied at each iteration—referred to as *inner residuals*—to support progressive token-wise refinement. An alternative design applies a single *outer residual* after all iterations, skipping intermediate feedback. Formally, this variant modifies the update rule as:

$$x^{(0)} = x,$$

$$x^{(t)} = \sum_{i=1}^{M} \hat{\mathbf{E}}_{i}(x^{(t-1)}) + \sum_{i=1}^{N} g_{t,i} \cdot \mathbf{E}_{i}(x^{(t-1)}), \quad t = 1, \dots, C,$$

$$y = x^{(C)} + x,$$

$$(9)$$

where only the final output receives residual feedback from the original input. As shown in Figure [7,](#page-7-1) this outer-residual-only design leads to slower convergence and higher final validation loss when trained on mathematical data. These results suggest that inner residuals play a key role in stabilizing multi-step reasoning, allowing more effective credit assignment and optimization along the iterative expert path.

Together, these results confirm that CoE's performance gains stem not only from multi-step routing, but also from the architectural mechanisms that support specialization via independent gating and stable refinement via inner residual connections.

#### <span id="page-7-0"></span>5.4 Co-activation Patterns Reveal Step-wise Expert Specialization

To further understand how communication steps affect routing behavior, we visualize the coactivation matrix between the two iterations of CoE. For each token routed to experts in iteration 0 and 1, we accumulate expert pair frequencies, resulting in a K × K co-activation matrix per layer. Figure [8](#page-8-1) shows the matrices across four layers trained and evaluated on MetaMathQA, and we present more settings for this expert co-activation in the Supplementary.

We observe that expert pairs are not uniformly distributed: each layer exhibits a diverse set of expert combinations across steps, suggesting that different iterations route to meaningfully different experts. This supports our hypothesis that CoE leverages iteration-specific gating to perform progressive, compositional refinement. Moreover, the sparsity and asymmetry of the co-activation matrices imply that the model is not simply repeating expert usage, but adapting routing based on intermediate representations.

<span id="page-8-1"></span>![](_page_8_Figure_0.jpeg)

Figure 8: Expert co-activation between communication steps. For each selected layer, we plot a matrix counting how often each expert pair (e0, e1) is activated across the two communication steps in CoE. The non-uniform, asymmetric patterns indicate that routing decisions differ meaningfully between steps, supporting the emergence of step-wise expert specialization.

Together with the results in Figure [7](#page-7-1) and Table [1,](#page-4-1) this analysis highlights that CoE achieves more than deeper computation—it enables structured multi-step specialization that standard MoE cannot capture.

#### <span id="page-8-0"></span>5.5 Theoretical Analysis: Combinatorial Flexibility and Effective Depth

We hypothesize the performance advantages of CoE could stem from two properties: combinatorial flexibility in expert selection and implicit depth expansion through expert communication. Conventional MoEs that select 2k experts in a single step, while CoE performs two separate top-k routing operations across iterations. This change increases the number of possible expert combinations from C(n, 2k) to C(n, k) 2 . For example, with n = 64 and k = 4, CoE leads to 823× more expert pairings than one-shot routing. It allows the model to encode a significantly more diverse set of expert interactions, which may improve its ability to route tokens in a more diverse manner.

In addition, CoE introduces iterative processing that deepens token representations over time. Since expert outputs from the first iteration influence routing in the second, the model applies distinct transformations across steps. A token may be refined by different experts or revisited by the same expert, enabling multi-pass representation refinement. This increases the model's effective depth without increasing parameter count or layer count. Recent analyses [\[27,](#page-11-3) [28,](#page-11-4) [29\]](#page-11-5) suggest that deeper internal computation pathways correlate with improved reasoning, particularly in math and logical inference. By enabling step-wise expert composition, CoE supports this depth-like refinement within a sparse, modular architecture.

# 6 Related Works

Mixture of Experts (MoE). MoE improves scaling efficiency by activating sparse subnetworks [\[30,](#page-11-6) [1\]](#page-9-0). Recent MoE-based LLMs such as Mixtral [\[10\]](#page-10-3), DeepSeek [\[6,](#page-9-5) [25,](#page-11-1) [31,](#page-11-7) [32\]](#page-11-8), Phi-3.5 [\[7\]](#page-10-0), and Qwen3 [\[9\]](#page-10-2) adopt top-k expert routing to reduce active parameter count during inference. These models maintain competitive performance with significantly lower memory cost. The architecture has also been adopted in Llama 4 [\[8\]](#page-10-1) and explored in OLMoE [\[11\]](#page-10-4) and Skywork [\[33\]](#page-11-9).

Iterative Architectures and Effective Depth. Instead of increasing width, iterative (recurrent) models reuse layers iteratively to expand effective depth [\[34,](#page-11-10) [35,](#page-12-0) [36\]](#page-12-1). Universal Transformers [\[19\]](#page-10-12) and ACT [\[37\]](#page-12-2) use input-adaptive computation steps; ALBERT [\[38\]](#page-12-3) and DEQ [\[39\]](#page-12-4) share or implicitly unroll layers. Recent work explores inference-time depth via latent recurrence [\[29\]](#page-11-5), offering computation-quality tradeoffs beyond prompt-based reasoning [\[40\]](#page-12-5). This setup improves reasoning controllability, while remaining robust under compression [\[41\]](#page-12-6).

Expert Reuse and Interaction. Recent work enhances expert capacity via iterative reuse. SparseUT [\[14\]](#page-10-7) and MoEUT [\[15\]](#page-10-8) combine MoE with depth-shared layers. RMoE [\[16\]](#page-10-9) introduces a recurrent routing state. OLMoE [\[11\]](#page-10-4) mixes global and token-wise expert selection. While most designs apply inter-layer reuse, intra-layer expert communication remains underexplored.

# 7 Conclusion

Chain-of-Experts (CoE) improves sparse model performance by introducing iterative expert processing within each transformer layer. Unlike traditional Mixture-of-Experts (MoE) architectures that treat expert computation as independent, CoE applies sequential routing with separate gating at each step, allowing experts to refine and exchange information across iterations. Empirical results show that CoE achieves lower validation loss, more efficient use of memory, and better compute scaling under matched parameter and runtime budgets. These findings support the view that sparse model quality depends not only on the number of experts used, but also on how information flows between them, pointing to a potentially better sparse foundation model architecture where modular reasoning depends on expert communication rather than solely expert capacity.

## 8 Limitations and Future Work

While Chain-of-Experts (CoE) demonstrates strong performance and efficiency, several practical and conceptual limitations remain. CoE introduces moderate time overhead in practice despite theoretical FLOP parity to MoE due to sequential processing. Selecting fewer experts per iteration reduces the degree of matrix multiplication parallelism, which can slow training on current hardware unless optimized at a low level. CoE also requires training from scratch and is not compatible with existing pretrained MoE checkpoints, limiting its use in transfer learning workflows. In addition, our implementation has only been tested on single-device setups. It remains unclear how CoE behaves under multi-node training or when used with newer training formats such as FP8.

Future work will focus on scaling model size, batch size, and training steps to examine whether CoE's advantages persist under standard scaling law conditions. Although our current experiments primarily use math datasets due to their challenging reasoning nature, broader domain evaluation is necessary. We will also study CoE's performance on downstream tasks, including language understanding and code generation benchmarks. Another direction is to increase the number of expert processing iterations per layer. Our current experiments use only two; the effect of deeper iterative depth remains unexplored. Finally, how to combine CoE with architectural variants that share experts across layers such as MoEUT remains an open-problem. These designs replace within-layer routing with global expert pools shared across the network, potentially enabling greater parameter efficiency and richer specialization patterns.

# References

- <span id="page-9-0"></span>[1] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeffrey Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *ICLR*, 2017.
- <span id="page-9-1"></span>[2] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-9-2"></span>[3] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*, 23(120):1–39, 2022.
- <span id="page-9-3"></span>[4] Barret Zoph. Designing effective sparse expert models. In *2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, pages 1044–1044, 2022. doi: 10.1109/IPDPSW55747.2022.00171.
- <span id="page-9-4"></span>[5] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, and James Laudon. Mixture-of-experts with expert choice routing, 2022. URL <https://arxiv.org/abs/2202.09368>.
- <span id="page-9-5"></span>[6] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models, 2024. URL <https://arxiv.org/abs/2401.06066>.

- <span id="page-10-0"></span>[7] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. *arXiv preprint arXiv:2404.14219*, 2024.
- <span id="page-10-1"></span>[8] Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation, 2025. URL <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>.
- <span id="page-10-2"></span>[9] Qwen Team. Qwen3: Think deeper, act faster, 2025. URL [https://qwenlm.github.io/](https://qwenlm.github.io/blog/qwen3/) [blog/qwen3/](https://qwenlm.github.io/blog/qwen3/).
- <span id="page-10-3"></span>[10] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-10-4"></span>[11] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. Olmoe: Open mixture-ofexperts language models. *arXiv preprint arXiv:2409.02060*, 2024.
- <span id="page-10-5"></span>[12] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of ´ experts, 2021. URL <https://arxiv.org/abs/2106.05974>.
- <span id="page-10-6"></span>[13] Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang, Wanqi Zhong, Wenhan Luo, Lin Ma, and Min Zhang. Uni-moe: Scaling unified multimodal llms with mixture of experts, 2024. URL <https://arxiv.org/abs/2405.11273>.
- <span id="page-10-7"></span>[14] Shawn Tan, Yikang Shen, Zhenfang Chen, Aaron Courville, and Chuang Gan. Sparse universal transformer. *arXiv preprint arXiv:2310.07096*, 2023.
- <span id="page-10-8"></span>[15] Robert Csord ´ as, Kazuki Irie, J ´ urgen Schmidhuber, Christopher Potts, and Christopher D Man- ¨ ning. Moeut: Mixture-of-experts universal transformers. *arXiv preprint arXiv:2405.16039*, 2024.
- <span id="page-10-9"></span>[16] Zihan Qiu, Zeyu Huang, Shuang Cheng, Yizhi Zhou, Zili Wang, Ivan Titov, and Jie Fu. Layerwise recurrent router for mixture-of-experts. *arXiv preprint arXiv:2408.06793*, 2024.
- <span id="page-10-10"></span>[17] Zihan Wang, Deli Chen, Damai Dai, Runxin Xu, Zhuoshu Li, and Y. Wu. Let the expert stick to his last: Expert-specialized fine-tuning for sparse architectural large language models, 2024. URL <https://arxiv.org/abs/2407.01906>.
- <span id="page-10-11"></span>[18] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.
- <span id="page-10-12"></span>[19] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. In *International Conference on Learning Representations*, 2019.
- <span id="page-10-13"></span>[20] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. [https://www.cerebras.net/blog/](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) [slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama), 2023. URL <https://huggingface.co/datasets/cerebras/SlimPajama-627B>.
- <span id="page-10-14"></span>[21] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models, 2024. URL <https://arxiv.org/abs/2309.12284>.
- <span id="page-10-15"></span>[22] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv:1803.05457v1*, 2018.
- <span id="page-10-16"></span>[23] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019. URL <https://arxiv.org/abs/1905.07830>.

- <span id="page-11-0"></span>[24] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language, 2019. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1911.11641) [1911.11641](https://arxiv.org/abs/1911.11641).
- <span id="page-11-1"></span>[25] DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma, Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024. URL <https://arxiv.org/abs/2405.04434>.
- <span id="page-11-2"></span>[26] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv: 2409.19256*, 2024.
- <span id="page-11-3"></span>[27] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: A theoretical perspective, 2023. URL [https:](https://arxiv.org/abs/2305.15408) [//arxiv.org/abs/2305.15408](https://arxiv.org/abs/2305.15408).
- <span id="page-11-4"></span>[28] William Merrill and Ashish Sabharwal. A little depth goes a long way: The expressive power of log-depth transformers, 2025. URL <https://arxiv.org/abs/2503.03961>.
- <span id="page-11-5"></span>[29] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. *arXiv preprint arXiv:2502.05171*, 2025.
- <span id="page-11-6"></span>[30] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. *Neural computation*, 3(1):79–87, 1991.
- <span id="page-11-7"></span>[31] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.
- <span id="page-11-8"></span>[32] Zihan Wang, Deli Chen, Damai Dai, Runxin Xu, Zhuoshu Li, and Y. Wu. Let the expert stick to his last: Expert-specialized fine-tuning for sparse architectural large language models, 2024. URL <https://arxiv.org/abs/2407.01906>.
- <span id="page-11-9"></span>[33] Tianwen Wei, Bo Zhu, Liang Zhao, Cheng Cheng, Biye Li, Weiwei Lu, Peng Cheng, Jianhao ¨ Zhang, Xiaoyu Zhang, Liang Zeng, et al. Skywork-moe: A deep dive into training techniques for mixture-of-experts language models. *arXiv preprint arXiv:2406.06563*, 2024.
- <span id="page-11-10"></span>[34] Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, and Tom Goldstein. End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and

- Kyunghyun Cho, editors, *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=PPjSKy40XUB>.
- <span id="page-12-0"></span>[35] Jay Bear, Adam Prugel-Bennett, and Jonathon Hare. Rethinking deep thinking: Stable learn- ¨ ing of algorithms using lipschitz constraints, 2024. URL [https://arxiv.org/abs/2410.](https://arxiv.org/abs/2410.23451) [23451](https://arxiv.org/abs/2410.23451).
- <span id="page-12-1"></span>[36] Sean Michael McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, and Tom Goldstein. Transformers can do arithmetic with the right embeddings. In *The Thirtyeighth Annual Conference on Neural Information Processing Systems*, 2024. URL [https:](https://openreview.net/forum?id=aIyNLWXuDO) [//openreview.net/forum?id=aIyNLWXuDO](https://openreview.net/forum?id=aIyNLWXuDO).
- <span id="page-12-2"></span>[37] Alex Graves. Adaptive computation time for recurrent neural networks. *arXiv preprint arXiv:1603.08983*, 2016.
- <span id="page-12-3"></span>[38] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In *International Conference on Learning Representations (ICLR)*, 2020.
- <span id="page-12-4"></span>[39] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In *Advances in Neural Information Processing Systems*, volume 32, 2019.
- <span id="page-12-5"></span>[40] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. *arXiv preprint arXiv:2201.11903*, 2022.
- <span id="page-12-6"></span>[41] Nan Zhang, Yusen Zhang, Prasenjit Mitra, and Rui Zhang. When reasoning meets compression: Benchmarking compressed large reasoning models on complex reasoning tasks. *arXiv preprint arXiv:2504.02010*, 2025.
- <span id="page-12-7"></span>[42] Jeffrey L. Elman. Finding structure in time. *Cognitive Science*, 14(2):179–211, 1990. ISSN 0364-0213. doi: https://doi.org/10.1016/0364-0213(90)90002-E. URL [https://www.](https://www.sciencedirect.com/science/article/pii/036402139090002E) [sciencedirect.com/science/article/pii/036402139090002E](https://www.sciencedirect.com/science/article/pii/036402139090002E).
- <span id="page-12-8"></span>[43] Jurgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic ¨ recurrent networks. *Neural Computation*, 4(1):131–139, 1992. doi: 10.1162/neco.1992.4.1. 131.
- <span id="page-12-9"></span>[44] Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ *Neural Comput.*, 9 (8):1735–1780, November 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL <https://doi.org/10.1162/neco.1997.9.8.1735>.
- <span id="page-12-10"></span>[45] Sho Takase and Shun Kiyono. Lessons on parameter sharing across layers in transformers. In Nafise Sadat Moosavi, Iryna Gurevych, Yufang Hou, Gyuwan Kim, Young Jin Kim, Tal Schuster, and Ameeta Agrawal, editors, *Proceedings of the Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)*, pages 78–90, Toronto, Canada (Hybrid), July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.sustainlp-1.5. URL <https://aclanthology.org/2023.sustainlp-1.5/>.
- <span id="page-12-11"></span>[46] Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-adaptive transformer. In *International Conference on Learning Representations (ICLR)*, 2020.
- <span id="page-12-12"></span>[47] Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout. In *International Conference on Learning Representations (ICLR)*, 2020.
- <span id="page-12-13"></span>[48] Dahyun Kim, Chanjun Park, Sanghoon Kim, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. Solar 10.7b: Scaling large language models with simple yet effective depth up-scaling, 2024. URL <https://arxiv.org/abs/2312.15166>.

- <span id="page-13-0"></span>[49] Robert Csord ´ as, Kazuki Irie, and Juergen Schmidhuber. The devil is in the detail: Simple ´ tricks improve systematic generalization of transformers. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 619–634, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.49. URL [https://aclanthology.org/2021.](https://aclanthology.org/2021.emnlp-main.49/) [emnlp-main.49/](https://aclanthology.org/2021.emnlp-main.49/).
- <span id="page-13-1"></span>[50] Juergen Schmidhuber. Self-delimiting neural networks, 2012. URL [https://arxiv.org/](https://arxiv.org/abs/1210.0118) [abs/1210.0118](https://arxiv.org/abs/1210.0118).

# A Extended Related Works

Mixture-of-Experts (MoE). MoE architectures extend neural capacity through conditional computation [\[30\]](#page-11-6), activating only a small subset of experts per input [\[1\]](#page-9-0). Modern MoE-based LLMs like Mixtral [\[10\]](#page-10-3) adopt top-2 routing across 8 feed-forward experts, reducing memory footprint during inference. This pattern enables models to scale parameter counts without increasing runtime cost. Several open models including DeepSeek [\[6,](#page-9-5) [25,](#page-11-1) [31,](#page-11-7) [32\]](#page-11-8), Phi-3.5 [\[7\]](#page-10-0), Qwen3 [\[9\]](#page-10-2), and Llama 4 [\[8\]](#page-10-1) demonstrate the practicality of MoE in production-grade systems. Complementary efforts such as OLMoE [\[11\]](#page-10-4) and Skywork [\[33\]](#page-11-9) also explore hybrid routing schemes and inference-time budget control.

Iterative Computation and Effective Depth. Iterative processing architectures offer an alternative scaling axis by reusing modules across time [\[34,](#page-11-10) [35,](#page-12-0) [36,](#page-12-1) [42\]](#page-12-7). Early work such as Schmidhuber [\[43\]](#page-12-8) and Hochreiter and Schmidhuber [\[44\]](#page-12-9) emphasized iteration (recurrence) as a means for structured memory and temporal abstraction. Modern models like Universal Transformers [\[19\]](#page-10-12) and Adaptive Computation Time (ACT) [\[37\]](#page-12-2) dynamically control depth through halting units, while ALBERT [\[38\]](#page-12-3) and Takase and Kiyono [\[45\]](#page-12-10) achieves parameter efficiency via cross-layer sharing. Implicit infinite-depth computation is enabled by deep equilibrium models (DEQ) [\[39\]](#page-12-4), and recent dynamic-depth strategies include early-exit [\[46\]](#page-12-11), LayerDrop [\[47\]](#page-12-12), and residual depth scaling [\[48\]](#page-12-13). More recently, latent-space recurrence has been explored as a lightweight mechanism for test-time refinement: Geiping et al. [\[29\]](#page-11-5) show that recursive hidden-state iteration improves math and logical reasoning. These methods internalize multi-step computation, in contrast to chain-of-thought prompting [\[40\]](#page-12-5). They also tend to be more compression-resilient [\[41\]](#page-12-6), consistent with findings on the robustness of recurrent representations [\[49,](#page-13-0) [50\]](#page-13-1).

Expert Reuse and Interaction. In addition to sparse activation, recent research focuses on improving expert utility through reuse and structural regularization. SparseUT [\[14\]](#page-10-7) and MoEUT [\[15\]](#page-10-8) integrate MoE into Universal Transformers, enabling both depth-sharing and expert reuse. MoEUT further unifies attention and FFN expert selection and adds normalization to stabilize cross-layer reuse. Layerwise recurrent MoE (RMoE) [\[16\]](#page-10-9) introduces a recurrent router state, enhancing contextaware routing decisions. DeepSeekMoE [\[6\]](#page-9-5) and OLMoE [\[11\]](#page-10-4) separates experts into global and token-routed groups to promote common feature sharing. However, these systems mostly adopt inter-layer sharing. Intra-layer expert communication remains relatively unexplored, opening opportunities for expert collaboration across time steps and interaction loops, an idea central to our Chain-of-Experts formulation.

## B Impact of Shared Experts

Our framework allows for flexible inclusion of shared experts across both CoE and MoE variants. To assess their utility, we conduct a controlled comparison using four configurations: CoE with and without a shared expert, and MoE with and without a shared expert. All these models are trained on the MetaMathQA dataset.

As shown in Table [2,](#page-15-0) introducing a shared expert leads to higher PIQA normalization scores. These gains are more pronounced in the smaller K=4, C=2 configuration, suggesting that shared experts help compensate for limited expert diversity by providing a stable backbone for generalization. On ARC-E and HellaSwag, however, performance remains largely similar, and in some cases models without shared experts slightly outperform their counterparts. It indicates that while shared experts can enhance performance on certain tasks, their benefit is not universal and may depend on task characteristics or expert routing depth.

## C Extended Training and Analysis

### C.1 Extending Training Steps

We extend training to 10,000 steps, 10% warmup to examine whether earlier conclusions hold under increased compute. Note that compute on MoE layers is approximately proportional to C × K,

<span id="page-15-0"></span>Table 2: Impact of shared expert on CoE performance across benchmarks. Using a shared expert yields marginal gains on PIQA for both MoE and CoE, while overall performance remains comparable across settings with and without shared experts.

| Setting               | ARC-E |       | HellaSwag |       | PIQA  |       |
|-----------------------|-------|-------|-----------|-------|-------|-------|
|                       | Acc   | Norm  | Acc       | Norm  | Acc   | Norm  |
| MetaMathQA (1 shared) |       |       |           |       |       |       |
| K=4, C=2              | 26.5% | 26.0% | 25.9%     | 26.3% | 54.5% | 51.4% |
| K=8, C=1              | 26.4% | 25.8% | 26.1%     | 26.0% | 54.6% | 52.3% |
| MetaMathQA (0 shared) |       |       |           |       |       |       |
| K=4, C=2              | 27.1% | 25.6% | 26.5%     | 26.0% | 52.2% | 51.6% |
| K=8, C=1              | 26.7% | 25.0% | 26.2%     | 26.5% | 54.0% | 52.2% |

representing the total number of expert invocations per token per layer. As shown in Figures [9](#page-16-0) and [10,](#page-17-0) all models eventually converge, but the scaling behavior differs.

In both the MetaMathQA (Figure [9\)](#page-16-0) and the general-domain SlimPajama settings (Figure [10\)](#page-17-0), CoE with K=8, C=2 consistently achieves a lower final validation loss than MoE with K=8, C=1. It indicates that increasing C in CoE offers a notable scaling factor. Additionally, CoE with K=4, C=2 converges faster than MoE early in training, though its final loss is similar. It suggests that CoE benefits more from compute scaling, especially in the high-capacity regime, and that increasing C is an effective lever for improving convergence without increasing total parameters or GPU memory requirements.

### C.2 Extending Communication Steps

To further explore the scaling behavior of CoE, we extend the communication steps C beyond the previously studied C = 2. Figures [9](#page-16-0) and [10](#page-17-0) report validation loss curves under fixed K = 8 and increasing C ∈ {2, 3, 4}. While increasing C from 1 to 2 yields a clear improvement, further increasing C shows diminishing or unstable gains, especially on the MetaMathQA benchmark.

Although CoE (K = 8, C = 3/4) extends computation within a fixed parameter space, the performance only matches or slightly outperforms the baseline in early training, and the final convergence is less robust. It suggests that naive scaling of C may lead to inefficient communication or unstable optimization dynamics, and may require more training compute to converge. Efficiently extending the communication horizon remains an open challenge, and we leave the development of more principled techniques for more effective expert communication to future work.

#### C.3 Extended Analysis on Residual Connections

In addition to the residual strategy proposed in Section [5.3,](#page-6-0) we explore an alternative residual design inspired by [\[29\]](#page-11-5), where the initial representation is added as a residual at every iteration. Formally, the update rule is defined as:

$$x^{(0)} = x,$$

$$x^{(t)} = \sum_{i=1}^{M} \hat{\mathbf{E}}_{i}(x^{(t-1)}) + \sum_{i=1}^{N} g_{t,i} \cdot \mathbf{E}_{i}(x^{(t-1)}) + x^{(0)}, \quad t = 1, \dots, C,$$

$$y = x^{(C)}.$$
(10)

We compare this variant (denoted as *Init*) with two other designs: adding residuals from the previous iteration (*Inner*) and from a separate outer loop representation (*Outer*). All methods are trained for 1000 steps on the MetaMathQA dataset. The loss comparison is presented in Table [3.](#page-16-1)

These results suggest that residual connections must be carefully designed. While the *Init* method provides some improvement over the *Outer* strategy, it still underperforms compared to *Inner* residual we used, which we adopt as the default in all main experiments.

<span id="page-16-1"></span>Table 3: Residual loss under different residual connection strategies.

|      | Inner Residual | Outer Residual | Init Residual |
|------|----------------|----------------|---------------|
| Loss | 1.12           | 1.21           | 1.18          |

<span id="page-16-0"></span>![](_page_16_Figure_2.jpeg)

Figure 9: Validation loss on MetaMathQA across 10,000 steps.

## D Analysis of Expert Co-Activation Visualizations

Each visualization in Figures [11](#page-18-0)[–22](#page-20-0) consists of four heatmaps, corresponding to the four transformer layers in our model. These heatmaps characterize expert transition behaviors across CoE iterations. Specifically, each heatmap presents a two-dimensional expert-to-expert transition matrix, where the x-axis denotes the *Next Expert* (i.e., experts assigned in iteration t + 1), and the y-axis represents the *Previous Expert* (i.e., experts assigned in iteration t). The intensity of each cell reflects the normalized frequency that tokens, previously processed by a given expert, are routed to another expert in the subsequent iteration at the same layer.

#### D.1 Effect of Dataset on Expert Co-Activation Patterns

We observe a consistent trend across datasets: expert transitions in general-domain corpora (SlimPajama) are more evenly distributed, whereas transitions in domain-specific data (Meta-MathQA) tend to be more concentrated (*e.g.*, Figure [17](#page-19-0) vs. Figure [15\)](#page-19-1). It suggests that experts in general-domain settings engage in a richer and more diverse interaction pattern, potentially due to the broader range of linguistic phenomena. Interestingly, we also find that both SlimPajama and MetaMathQA exhibits increasing expert specialization in the final layers compared to the early layers, as evidenced by the emergence of dominant transition paths in the last one or two layers.

#### D.2 Temporal Dynamics of Expert Co-Activation During Training

On the general-domain dataset SlimPajama, we observe that expert transitions become progressively more concentrated as training proceeds—particularly in deeper layers such as layer 3 (*cf.* Figure [17](#page-19-0) vs. Figure [18\)](#page-19-2). As the model is exposed to more diverse textual patterns, it gradually converges on a smaller subset of routing pathways that are reused consistently. This is reflected visually in the heatmaps: bright regions grow brighter, and dark regions become darker, indicating increasing polarization in expert assignment. Such a trend suggests that the model is identifying persistent and efficient expert transitions that generalize across large-scale linguistic contexts. In effect, the routing structure simplifies over time—emphasizing a few highly specialized paths that dominate token flow in later layers.

<span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

Figure 10: Validation loss on general-domain SlimPajama across 10,000 steps.

In contrast, on the more focused domain of MetaMathQA, expert transitions begin with a relatively narrow distribution—most tokens are routed similarly across iterations, resulting in sharp, localized activation patterns in early training (*cf.* Figure [15\)](#page-19-1). However, as training progresses, these patterns become more diffuse (*cf.* Figure [16\)](#page-19-3), reflecting a growing diversity in how tokens are routed across experts. This divergence likely arises because the model initially relies on uniform strategies when the domain is small and predictable, but later discovers the benefits of assigning different reasoning problems to different expert transitions. That is, instead of collapsing to fixed pipelines, the model in MetaMathQA explores increasingly differentiated expert flows as it learns to segment mathematical tasks by latent structure.

#### D.3 Intra-Layer Expert Flow and Role Differentiation

Within each layer, the diagonal intensity remains relatively low, indicating that experts tend not to reprocess tokens they previously handled. It affirms the flowing nature of CoE, where tokens are progressively transformed by distinct experts. However, we do observe a clear asymmetry between rows and columns in some matrices. Specifically, certain experts frequently act as entry points—handling more tokens in earlier iterations (brighter rows), while others serve as accumulators—processing aggregated representations in later iterations (brighter columns). This role differentiation is particularly pronounced on MetaMathQA (see Figure [20\)](#page-20-1), suggesting emergent task-driven specialization among experts.

### D.4 Comparison with MoE: Stability of Expert Usage

Unlike MoE, which often suffers from expert dropout or collapse in deeper layers during late training stages, CoE maintains robust expert utilization. Specifically, we find that in CoE, each expert tends to be predominantly used in either the first or second layer, but rarely vanishes entirely. This distribution alleviates the expert collapse issue observed in MoE [\[3\]](#page-9-2), i.e., the CoE routing scheme could naturally induce expert usage diversity and layer-specific specialization.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 11: Layer-wise routing pattern on MetamathQA under the 4-experts-per-iteration, 2-iteration setup (2k training steps).

![](_page_18_Figure_2.jpeg)

Figure 12: Layer-wise routing pattern on MetamathQA under the 4-experts-per-iteration, 2-iteration setup (10k training steps).

![](_page_18_Figure_4.jpeg)

Figure 13: Layer-wise routing pattern on SlimPajama under the 4-experts-per-iteration, 2-iteration setup (2k training steps).

![](_page_18_Figure_6.jpeg)

Figure 14: Layer-wise routing pattern on SlimPajama under the 4-experts-per-iteration, 2-iteration setup (10k training steps).

<span id="page-19-1"></span>![](_page_19_Figure_0.jpeg)

Figure 15: Layer-wise routing pattern on MetamathQA under the 8-experts-per-iteration, 2-iteration setup (2k training steps).

<span id="page-19-3"></span>![](_page_19_Figure_2.jpeg)

Figure 16: Layer-wise routing pattern on MetamathQA under the 8-experts-per-iteration, 2-iteration setup (10k training steps).

<span id="page-19-0"></span>![](_page_19_Figure_4.jpeg)

Figure 17: Layer-wise routing pattern on SlimPajama under the 8-experts-per-iteration, 2-iteration setup (2k training steps).

<span id="page-19-2"></span>![](_page_19_Figure_6.jpeg)

Figure 18: Layer-wise routing pattern on SlimPajama under the 8-experts-per-iteration, 2-iteration setup (10k training steps).

![](_page_20_Figure_0.jpeg)

Figure 19: Layer-wise routing pattern on MetamathQA under the 8-experts-per-iteration, 1-iteration setup (2k training steps).

<span id="page-20-1"></span>![](_page_20_Figure_2.jpeg)

Figure 20: Layer-wise routing pattern on MetamathQA under the 8-experts-per-iteration, 1-iteration setup (10k training steps).

![](_page_20_Figure_4.jpeg)

Figure 21: Layer-wise routing pattern on SlimPajama under the 8-experts-per-iteration, 1-iteration setup (2k training steps).

<span id="page-20-0"></span>![](_page_20_Figure_6.jpeg)

Figure 22: Layer-wise routing pattern on SlimPajama under the 8-experts-per-iteration, 1-iteration setup (10k training steps).