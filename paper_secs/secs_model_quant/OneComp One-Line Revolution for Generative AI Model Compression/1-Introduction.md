# 1 Introduction

Large-scale models achieve strong performance; however, they incur substantial memory and computational overhead, hindering practical deployment [\(Chen et al.,](#page-9-0) [2023\)](#page-9-0). These constraints are particularly stringent for edge devices. To bridge the gap between accuracy and deployability, prior work has explored compression techniques such as quantization [\(Lang et al.,](#page-9-1) [2024;](#page-9-1) [Gong et al.,](#page-9-2) [2024\)](#page-9-2), pruning [\(Wang et al.,](#page-10-0) [2024;](#page-10-0) [Cheng et al.,](#page-9-3) [2024\)](#page-9-3), low-rank adaptation [\(Yang et al.,](#page-10-1) [2024;](#page-10-1) [Hu](#page-9-4) [et al.,](#page-9-4) [2022\)](#page-9-4), and knowledge distillation [\(Xu et al.,](#page-10-2) [2024a\)](#page-10-2). Among these techniques, layer-wise posttraining quantization (PTQ) is one of the most practical and widely adopted methods for largescale LLMs [\(Frantar et al.,](#page-9-5) [2022;](#page-9-5) [Lin et al.,](#page-9-6) [2024;](#page-9-6) [Yao et al.,](#page-10-3) [2022;](#page-10-3) [Chee et al.,](#page-9-7) [2023\)](#page-9-7). By focusing on individual linear layers, layer-wise PTQ simplifies the problem to least-squares estimation

of linear transformations, which enables for efficient solvers and straightforward implementations. Despite its simplicity, layer-wise PTQ delivers strong empirical performance and is used both as a standalone method and as an initialization for more complex block-wise or global PTQ frameworks [\(Malinovskii et al.,](#page-10-4) [2024;](#page-10-4) [Guan et al.,](#page-9-8) [2024\)](#page-9-8).

However, the layer-wise structure imposes significant limitations. Classical layer-wise PTQ methods, such as GPTQ [\(Frantar et al.,](#page-9-5) [2022\)](#page-9-5), AWQ [\(Lin et al.,](#page-9-6) [2024\)](#page-9-6), and QuIP [\(Chee et al.,](#page-9-7) [2023\)](#page-9-7), optimize each linear layer independently by utilizing input activations. Consequently, these methods are confined to activation-aware weight approximation and indirectly affect model-level outputs; quantization error may become significant. Recent work, including QEP [\(Arai and Ichikawa,](#page-8-0) [2025\)](#page-8-0) and GPTAQ [\(Li et al.,](#page-9-9) [2025\)](#page-9-9), relaxes this constraint by modifying the layer-wise objective to address the mismatch between pre- and postquantization activations. This modification enables controlled error propagation across layers, preserving the sequential pipeline and reducing the accumulation of quantization errors. LoaQ [\(Lin and](#page-9-10) [Wan,](#page-9-10) [2025\)](#page-9-10) further extends the target from linear layers to specific submodules by explicitly approximating the outputs of residual connections and RMSNorm. This approach aligns layer-wise PTQ more closely with the behavior at the model-level. Nevertheless, these advances remain specialized: QEP is still anchored in individual linear layers, and LoaQ is limited to specific submodules; they do not provide a unified treatment of more general submodules, activation, or KV-cache quantization.

This study goes beyond traditional layer-wise formulations by introducing a framework for submodule quantization that preserves the standard layer-wise pipeline. We propose Layer-Projected Coordinate Descent (LPCD), which optimizes arbitrary submodules in the output space and projects the relaxed solutions back into the quantization do-

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: Output MSE across Transformer blocks in Llama 3 8B with 4, 3, and 2 bit weight quantization. LPCD consistently yields lower quantization error than QEP and LoaQ.

main using existing layer-wise PTQ algorithms as *layer projectors*. This perspective unifies classical layer-wise PTQ, QEP, and LoaQ as special cases while extending naturally to more general submodules, activations, and KV caches. We instantiate LPCD on grouped-query KV, VO aggregation, and MLP up-down blocks, demonstrating that LPCD significantly reduces quantization error compared to QEP and LoaQ, as shown in Figure 1. Extensive experiments across diverse tasks and bit-widths demonstrate that LPCD-based submodule quantization consistently enhances both layer-wise PTQ methods and existing submodule approaches. Our main contributions are as follows:

- LPCD is a *unified framework* that extends layer-wise PTQ to *arbitrary* submodules by reusing existing layer-wise quantizers.
- An effective method for quantizing coherent Transformer submodules reduces quantization error while remaining fully compatible with layer-wise PTQ pipelines.
- LPCD consistently enhances both layer-wise PTQ and existing submodule methods across diverse bit-widths and models, while avoiding unstable STE heuristics (Bengio et al., 2013).

## 2 Related Work

Quantization for LLMs includes data-free PTQ, layer-wise PTQ, block or global PTQ, and QAT (Dettmers and Zettlemoyer, 2023; Frantar et al., 2022; Tseng et al., 2024; Egiazarian et al., 2024; Xu et al., 2024b; Wang et al., 2023; Liu et al., 2023). Among these, weight-only layer-wise PTQ is particularly popular because each linear layer reduces to a least-squares problem that can be efficiently quantized on a single GPU (Frantar et al.,

2022; Lin et al., 2024; Chee et al., 2023; Zhao et al., 2025). Layer-wise PTQ methods are typically implemented as compensation-based schemes, such as GPTQ and its variants (Frantar et al., 2022; Behdin et al., 2023; Liu et al., 2024a; Guan et al., 2024), rotation-based schemes, such as QuIP and its successors (Chee et al., 2023; Tseng et al., 2024; Liu et al., 2024b; Ashkboos et al., 2024), or salience-based schemes, such as AWQ and mixedprecision methods (Dettmers et al., 2022, 2023b; Shang et al., 2023; Lin et al., 2024). More recently, QEP (Arai and Ichikawa, 2025), GPTAQ (Li et al., 2025), and Oronos (Zhang et al., 2025) refine the layer-wise loss by compensating for mismatches between pre- and post-quantization activations at the linear-layer level. Meanwhile, LoaQ extends this concept to sub-layers, including residual connections and RMSNorm (Lin and Wan, 2025). Our work follows this direction but provides a unified framework that applies layer-wise PTQ operators to minimize losses over more general submodules, such as KV, VO, and MLP, extending beyond individual layers or a fixed set of sub-layers.

## <span id="page-1-1"></span>3 Preliminaries

#### 3.1 Notation

In this section, we introduce the notation used consistently throughout the paper. Let  $[R] := \{1,\dots,R\}$  be a fixed finite index set. The expression  $\|\cdot\|_F$  denotes the Frobenius norm. The set  $\mathbb{Q} \subset \mathbb{R}$  denotes a fixed finite set of quantization levels, e.g., the grid induced by a b-bit quantizer, which we refer to as a b-bit quantization scheme. Let  $W \in \mathbb{R}^{N \times M}$  and  $\widehat{W} \in \mathbb{Q}^{N \times M}$  denote the original and quantized weight matrices. similarly, let  $X \in \mathbb{R}^{T \times N}$  and  $\widehat{X} \in \mathbb{R}^{T \times N}$  represent the activation matrices before and after quantization, where

T is the number of tokens and N is the feature dimension.  $R \in \mathbb{R}^{T \times M}$  and  $\widehat{R} \in \mathbb{R}^{T \times M}$  denote the full-precision and quantized residual streams, respectively. We define the following  $N \times N$  matrices that arise in the layer-wise quantization loss:  $H = X^{\top}X$  and  $\widehat{H} = \widehat{X}^{\top}\widehat{X}$ , which represent the Hessian matrix of quantization error;  $I_N$  the  $N \times N$  identity matrix.

### 3.2 Layer-wise PTQ

In PTQ, it is commonly assumed that accurately approximating the weights of each linear layer is sufficient to preserve overall model performance; thus, layer-wise PTQ treats each layer as an independent optimization problem. Layer-wise PTQ typically quantizes layers sequentially in a single forward pass, storing only the inputs of the current layer, which significantly reduces memory usage and computational demand. Unlike blockor model-wise schemes that must manage nonlinearities and rely on pseudo-gradients, e.g., STE (Bengio et al., 2013), layer-wise PTQ circumvents these issues and is consequently often employed as an initializer for more comprehensive methods. Consequently, the final performance of most PTQ pipelines is largely determined by the quality of the underlying layer-wise quantization. In the weightonly setting, classical layer-wise objectives decompose into two independent subproblems.

**Direct Weight Quantization.** Direct weight approximation quantizes the model parameters W through the following projection:

$$\widehat{W} = \Pi_{\mathbb{Q}}^{(d)}(W) \coloneqq \underset{\widehat{W}}{\operatorname{argmin}} \|\widehat{W} - W\|_F^2, \quad (1)$$

where the objective seeks the optimal quantized value for each weight independently. A representative method within this framework is the Round-To-Nearest (RTN) technique, which quantizes each weight by rounding it to the closest point in the quantization grid.

Activation-Aware Quantization. Activation-aware quantization quantizes the model parameters W by addressing the following activation-aware, layer-wise optimization problem:

$$\widehat{W} = \Pi^{(a)}_{\mathbb{Q}}(W) \coloneqq \operatorname*{argmin}_{\widehat{W}} \|\widehat{X}(\widehat{W} - W)\|_F^2, \ (2)$$

where the objective explicitly incorporates the input activation distribution, thereby reducing the effect of quantization error on the output of the linear layer. A typical approach in this category is GPTQ, which leverages the second-order information H of this objective, precomputed and cached for reuse, to further reduce output distortion during quantization. Leading layer-wise PTQ methods employ distinct optimization strategies to minimize Eq. (2). For example, GPTQ quantizes parameters row-wise by sequentially minimizing reconstruction error and propagating residual corrections to the remaining unquantized entries until each row is fully quantized. AWQ (Lin et al., 2024) identifies a small subset of *salient weights* whose magnitudes significantly influence the layer outputs and rescales these weights before quantization.

### 3.3 Quantization Error Propagation

Recent work demonstrates that the basic layer-wise PTQ formulation can be improved by explicitly considering the discrepancy between pre- and post-quantization activations. QEP is a representative method in this field of study. Rather than minimizing only the activation-aware loss in Eq. (2), QEP directly approximates the outputs of the linear layer by solving

<span id="page-2-2"></span>
$$\widehat{W}^{\text{QEP}} = \underset{\widehat{W}}{\operatorname{argmin}} \|\widehat{X}\widehat{W} - XW\|_F^2.$$
 (3)

As shown in QEP, this objective is equivalent to

$$\widehat{W}^{\text{QEP}} = \operatorname*{argmin}_{\widehat{W}} \|\widehat{X}(\widehat{W} - W^*)\|_F^2.$$

<span id="page-2-1"></span>where  $W^*=(I_N+\widehat{H}^{-1}C)W$  is accompanied by  $C=\widehat{X}^\top(X-\widehat{X})$ , which represents the error propagation matrix. Thus, QEP is first implemented by forming the corrected target  $W^*=(I_N+\widehat{H}^{-1}C)W$  and then applying a standard activation-aware layer-wise quantizer to  $W^*$ . To control for overfitting on the calibration set, QEP introduces a tunable coefficient  $\alpha\in[0,1]$  and employs  $W^*(\alpha)=(I_N+\alpha\widehat{H}^{-1}C)W$ , which interpolates between the original weights  $\alpha=0$  and the fully corrected weights. This simple modification consistently outperforms conventional layer-wise PTO.

### <span id="page-2-0"></span>3.4 Residual Path Quantization

Beyond individual linear layers, modern Transformers are governed by complex submodules that incorporate multiple linear projections, nonlinear functions, and residual connections. LoaQ extends

the QEP formulation to such submodules, targeting Transformer submodules that consist of a self-attention block, an MLP block, a residual connection, and RMSNorm. LoaQ directly matches the residual stream by minimizing

$$\widehat{W}^{\operatorname{LoaQ}} = \operatorname*{argmin}_{\widehat{W}} \| (\hat{R} + \widehat{X} \widehat{W}) - (R + XW) \|_F^2,$$

where R and  $\widehat{R}$  denote the full-precision and quantized residual streams, respectively.

As shown in LoaQ, this objective can be expressed in the same activation-aware form as Eq. (2) with a corrected target  $W^*(\alpha, \beta)$ ; the original LoaQ formulation corresponds to using the full corrections with  $\alpha = \beta = 1$ .

$$\widehat{W}^{\text{LoaQ}} = \underset{\widehat{W}}{\operatorname{argmin}} \|\widehat{X}(\widehat{W} - W^*(\alpha, \beta))\|_F^2, (4)$$

$$W^*(\alpha, \beta) = (I + \alpha \widehat{H}^{-1}C)W + \beta \widehat{H}^{-1}\Gamma$$

where  $\Gamma=\widehat{X}^{\top}(R-\widehat{R})$  is the residual-path correction and  $\alpha,\beta\in[0,1]$  are tunable parameters that enhance stability and prevent over-correction. Thus, LoaQ replaces the PTQ target with a corrected term that includes both the linear-layer correction  $(I_N+\widehat{H}^{-1}C)W$  and the residual-path correction  $\widehat{H}^{-1}\Gamma$ . Since standard Transformer blocks apply normalization, such as RMSNorm, after each submodule, the representations used by downstream layers are the *normalized* outputs. Accordingly, LoaQ introduces an additional objective that aligns explicitly with these normalized submodule outputs:

$$\min_{\widehat{W} \in \mathbb{Q}^{N \times M}} \|\operatorname{Norm}(\widehat{R} + \widehat{X}\widehat{W}) - \operatorname{Norm}(R + XW)\|_F^2,$$

where  $\operatorname{Norm}(\cdot)$  denotes normalization. To keep the problem tractable, LoaQ freezes the RMSNorm scaling factors, treating the normalization matrices as fixed, precomputed functions of the inputs so that the dependence on  $\widehat{W}$  remains linear. Under this mild approximation, LoaQ demonstrates that the aforementioned objective can be rewritten in the same activation-aware form as in Eq. (4), thereby enabling the direct use of existing layer-wise PTQ solvers while aligning normalized submodule outputs across more complex submodules.

### 4 Method

This section introduces LPCD, a unified framework for quantizing arbitrary submodules that extends layer-wise PTQ beyond these limited scenarios. We first provide a general, architecture-agnostic formulation. We then demonstrate that suitable choices of blocks and losses recover QEP and LoaQ as special cases. Finally, we instantiate LPCD on concrete Transformer submodules, deriving update rules for the QK, VO, and MLP up-down blocks.

### 4.1 Problem Setting

We consider the following collection of block variables:

$$\widehat{M}_r \in \mathbb{Q}^{N_r \times K_r}, \ r \in [R].$$

These variables are subject to quantization and represent weight matrices, activation matrices, or, more generally, intermediate model representations such as KV caches. Let

<span id="page-3-0"></span>
$$L: \prod_{r=1}^{R} \mathbb{R}^{N_r \times K_r} \to \mathbb{R},$$
 
$$(M_1, \dots, M_R) \mapsto L(M_1, \dots, M_R)$$

be a loss function that quantifies the discrepancy between a full-precision reference model and its quantized counterpart. The discrete quantization problem is defined as follows:

<span id="page-3-1"></span>
$$\min_{\widehat{M}_1,\dots,\widehat{M}_R} L(\widehat{M}_1,\dots,\widehat{M}_R).$$
(5)

Since  $\mathbb Q$  is finite, the feasible set of Eq. (5) is a subset of  $\prod_{r=1}^R \mathbb R^{N_r \times K_r}$ . In particular, this feasible set is discrete and non-convex, which makes the direct global optimization of Eq. (5) generally intractable. In this work, we aim to generalize our formulation to encompass both weights and activations by partitioning the index set [R] into two disjoint subsets,  $\mathcal R_w$  and  $\mathcal R_a$ , such that  $\mathcal R_w \cup \mathcal R_a = [R]$  and  $\mathcal R_w \cap \mathcal R_a = \emptyset$ . When  $r \in \mathcal R_w$ , the block  $M_r$  represents the weight matrix of a linear layer, whereas when  $r \in \mathcal R_a$ , the block  $M_r$  denotes the activation matrix.

### 4.2 Layer-Projected Coordinate Descent

We introduce the LPCD to approximately solve Eq. (5). For each  $r \in [R]$  and outer iteration  $t \in \mathbb{N}$ , let  $\widehat{M}_r^{(t)} \in \mathbb{Q}^{N_r \times K_r}$  represent the quantized block r during the t-th outer iteration. We consider a cyclic block-coordinate scheme that updates the blocks in the order  $r \in [R]$  and then repeats. At outer iteration t, when updating block r, we keep all other blocks fixed at their most recently updated

quantized values. Formally, we define the blockwise objective

$$\begin{split} L_r^{(t)}(U) &\coloneqq \\ L(\widehat{M}_1^{(t)}, \dots, \widehat{M}_{r-1}^{(t)}, U, \widehat{M}_{r+1}^{(t-1)}, \dots, \widehat{M}_R^{(t-1)}), \end{split}$$

where  $U \in \mathbb{R}^{N_r \times K_r}$  is the relaxed variable corresponding to  $\widehat{M}_r$  at iteration t. For s < r, we use the current-iteration values  $\widehat{M}_s^{(t)}$ ; for s > r, we use the values from the previous-iteration  $\widehat{M}_s^{(t-1)}$ . Adapting the sequence of block updates presents an intriguing direction for future research. The update of block  $r \in [R]$  is conducted in two stages.

**Relaxation Step.** We first relax the quantization constraint and solve the corresponding unconstrained continuous optimization problem

$$\overline{M}_r^{(t)} = \underset{U}{\operatorname{argmin}} L_r^{(t)}(U). \tag{6}$$

Whenever  $L_r$  has a structure that allows for a closed-form minimizer, such as when  $L_r$  is a strictly convex quadratic function of U, we compute the unique minimizer analytically and adopt this as  $\overline{M}_r^{(t)}$ . Otherwise, we approximate a minimizer, or at least a stationary point of Eq. (6), by applying a numerical optimization method, such as gradient descent or an accelerated first-order scheme, to the differentiable function. Under standard regularity assumptions, such as the Lipschitz continuity of  $\nabla L_r$  and suitable step-size conditions, these methods converge to a stationary point of  $L_r$ . In either case, this step produces a continuous candidate  $\overline{M}_r^{(t)} \in \mathbb{R}^{N_r \times K_r}$  that is either exactly or approximately optimal for block r, conditional on the other blocks being fixed at their current quantized values.

**Projection Step.** In this step, we reapply the quantization constraint by projecting  $\overline{M}_r^{(t)}$  onto  $\mathbb{Q}^{N_r \times K_r}$  through layer-wise PTQ projections. If  $r \in \mathcal{R}_w$  is a weight matrix, we apply a layer-wise PTQ projection defined in Eq. (1) or Eq. (2). Concretely, we select either the direct weight projection  $\Pi^{(d)}_{\mathbb{Q}}$  or the activation-aware projection  $\Pi^{(a)}_{\mathbb{Q}}$  and define

$$\widehat{M}_r^{(t+1)} = \Pi_{\mathbb{Q}}^{(w)}(\overline{M}_r^{(t)}), \ \Pi_{\mathbb{Q}}^{(w)} \in \{\Pi_{\mathbb{Q}}^{(d)}, \Pi_{\mathbb{Q}}^{(a)}\}.$$

If  $r \in \mathcal{R}_a$  is an activation matrix, it is quantized by applying the direct projection  $\Pi^{(d)}_{\mathbb{Q}}$  to  $\overline{M}^{(t)}_r$ .

$$\widehat{M}_r^{(t+1)} = \Pi_{\mathbb{Q}}^{(d)}(\overline{M}_r^{(t)}), \ r \in \mathcal{R}_{\mathbf{a}}.$$

This choice aims to minimize the entrywise distortion of the Frobenius norm between the full-precision and quantized activations at block r.

By alternating these relaxation and projection steps, each block update maps a feasible quantized tuple  $(\widehat{M}_1^{(t)},\ldots,\widehat{M}_R^{(t)})$  to another feasible tuple  $(\widehat{M}_1^{(t+1)},\ldots,\widehat{M}_R^{(t+1)})$  that continues to satisfy the constraints of Eq. (5). Since  $\mathbb Q$  is finite and the projections are well defined, feasibility is preserved at each iteration.

