# LPCD: Unified Framework from Layer-Wise to Submodule Quantization

Yuma Ichikawa

Fujitsu Limited RIKEN Center for AIP

Yudai Fujimoto Fujitsu Limited Institute of Science Tokyo

Akira Sakai Fujitsu Limited Tokai University

# Abstract

Post-training quantization (PTQ) aims to preserve model-level behavior; however, most methods focus on individual linear layers. Even recent extensions, such as QEP and LoaQ, which mitigate error propagation or target specific submodules, still rely on layer-wise formulations and fail to capture the behavior of larger submodules. We introduce Layer-Projected Coordinate Descent (LPCD), a unified framework that extends PTQ beyond layers by optimizing relaxed objectives across arbitrary submodules and projecting the solutions with standard layer-wise quantizers. LPCD generalizes existing methods and provides a principled approach to quantizing complex submodules while maintaining the efficiency and compatibility of layer-wise PTQ pipelines. Across diverse LLM architectures and bit-widths, LPCDbased submodule quantization consistently enhances both layer-wise PTQ methods and existing submodule approaches.

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

# <span id="page-4-3"></span>4.3 QEP as a Single LPCD Update

<span id="page-4-0"></span>This section shows that a single iteration of LPCD corresponds to both QEP and LoaQ. For brevity, we explicitly demonstrate that QEP corresponds to one LPCD iteration; an analogous argument can be used to prove the LoaQ case. To relate to QEP, we consider a two-block instance of the general formulation with the block variables  $M_1 = \widehat{W} \in \mathbb{R}^{N \times M}$  and  $M_2 = \widehat{X} \in \mathbb{R}^{T \times N}$ . The global objective is given by

<span id="page-4-1"></span>
$$L(\widehat{W}, \widehat{X}) = \|\widehat{X}\widehat{W} - XW\|_F^2. \tag{7}$$

In this case, the following proposition holds:

**Proposition 4.1.** Consider the objective defined in Eq. (7) with blocks  $M_1 = \widehat{W}$  and  $M_2 = \widehat{X}$ . Fix the activation block  $\widehat{X}$  and perform a single LPCD update on the weight block  $\widehat{W}$ . Let  $\widehat{W}^{(1)}$  denote the value of the weight block following this single LPCD update. Then  $\widehat{W}^{\text{QEP}} = \widehat{W}^{(1)}$ .

*Proof.* The LPCD update of the weight block  $M_1 = \widehat{W}$  consists of two steps:

**Relaxation Step.** Fix  $\widehat{X}$ . In the first outer iteration, the objective for  $\widehat{W}$  takes the following form:

$$L_1^{(1)}(U) = \|\widehat{X}^{(0)}U - XW\|_F^2,$$

where  $\widehat{X}^{(0)} = \widehat{X}$ . Assume that  $\widehat{H}$  is invertible. The optimality condition leads to the following minimizer:

$$\overline{W}^{(1)} = \operatorname*{argmin}_{U} L_1^{(1)}(U) = (I_N + \widehat{H}^{-1}C)W.$$

**Projection Step.** For a weight block, LPCD employs a layer-wise PTQ projection using either the direct or activation-aware projection,  $\Pi^{(d)}_{\mathbb{Q}}$  or  $\Pi^{(a)}_{\mathbb{Q}}$ , as defined in Section 3. In the QEP setting, the activation-aware projector  $\Pi^{(a)}_{\mathbb{Q}}$  is employed. The projection step, therefore, reads

<span id="page-4-2"></span>
$$\widehat{W}^{(1)} = \Pi_{\mathbb{Q}}^{(a)} \left( \overline{W}^{(1)} \right) = \Pi_{\mathbb{Q}}^{(a)} \left( (I_N + \widehat{H}^{-1}C)W \right). \tag{8}$$

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 2: Conceptual diagram of the submodules considered in this work using LPCD; the regions enclosed by the red dashed boxes correspond to submodules.

Finally, note that the QEP solution  $\widehat{W}^{\text{QEP}}$  from Eq. (3) satisfies

$$\begin{split} \widehat{W}^{\text{QEP}} &= \underset{\widehat{W}}{\operatorname{argmin}} \, \| \widehat{X} \widehat{W} - XW \|_F^2 \\ &= \underset{\widehat{W}}{\operatorname{argmin}} \, \| \widehat{X} \big( \widehat{W} - \overline{W}^{(1)} \big) \|_F^2. \end{split}$$

By definition of  $\Pi^{(a)}_{\mathbb{Q}}$ , the right-hand side coincides with the projection in Eq. (8). Therefore,  $\widehat{W}^{\text{QEP}} = \widehat{W}^{(1)}$  as stated.

**Remark 4.2.** An analogous argument demonstrates that LoaQ can also be interpreted as a single LPCD update for a suitably extended objective. For example, by considering

$$L(\widehat{W}, \widehat{X}, \widehat{R}) = \|(\widehat{R} + \widehat{X}\widehat{W}) - (R + XW)\|_F^2,$$

the same relaxation-projection decomposition yields  $\widehat{W}^{LoaQ} = \widehat{W}^{(1)}$ , with  $\widehat{W}^{(1)}$  computed by LPCD on this augmented submodule.

Moreover, LPCD allows us to extend single-step algorithms that compensate for quantization error, such as QEP, to various settings. Specifically, the extension of QEP to activation quantization is described in Appendix B.1; its extension to KV-cache quantization is outlined in Appendix B.2; and its extension to preprocessing using rotation matrices is detailed in Appendix B.3. All these algorithms can be implemented within a layer-wise PTQ framework, significantly reducing memory costs during quantization while minimizing quantization error. Furthermore, both the QEP, viewed as a single-step method, and its extensions are expected to achieve higher performance by increasing the number of iterations in the alternating optimization process.

## <span id="page-5-1"></span>4.4 Submodule PTQ

In this study, we apply LPCD to several submodules for which the relaxation step provides a closed-form solution. Although closed-form expressions exist, some of these problems are memory-inefficient to solve exactly; therefore, so we approximate the relaxation step in practice; see Appendix A.1. A conceptual diagram of this procedure is presented in Figure 2. In the following, we briefly describe how the proposed method is applied to each submodule. Note that the proposed method can also be applied to submodules with nonlinear transformations by approximating the minimization; a more exhaustive study of such applications is left for future work.

**QK Module.** We consider a setting in which grouped-query attention is quantized at the level of its QK submodule. Specifically, a single key and value are shared within each group of *G* heads. The objective of the QK module is expressed as

$$L(\hat{W}_{Q}, \hat{W}_{K}) = \sum_{h \in [H]} \|M \odot (\widehat{S}^{(h)} - S^{(h)})\|_{F}^{2}$$
$$\widehat{S}^{(h)} = \mathcal{R}(\widehat{X}\widehat{W}_{Q}^{(h)})\mathcal{R}(\widehat{X}\widehat{W}_{K}^{(g)})^{\top},$$
$$S^{(h)} = R(XW_{Q}^{(h)})R(XW_{K}^{(g)})^{\top}.$$

where  $g=\lfloor h-1/G\rfloor$  and  $M\in\{0,1\}^{T\times T}$  are binary upper-triangular matrices that represent the causal mask. The operator  $\mathcal{R}(\cdot)$  denotes rotary positional encoding (RoPE). When either  $\widehat{W}_Q$  or  $\widehat{W}_K$  is fixed, the objective reduces to a linear least-squares formulation, analogous to QEP. The detailed update rule for the relaxation step is provided in Appendix A.1.1

**VO Module.** Next, we consider the submodule that aggregates the attention scores  $S^{(h)} = \operatorname{Softmax}(S^{(h)})$  obtained by applying the softmax function after the KV module, and we quantize this component. Specifically, the objective can be

<span id="page-6-0"></span>

| Table 1: Perplexity (↓) on WikiText-2 for LLaMA and Qwen models across different bit-widths and quantization |
|--------------------------------------------------------------------------------------------------------------|
| methods.                                                                                                     |

| Bits | Method |      | LLaMA2-7B | LLaMA2-13B | LLaMA3-8B | Qwen3-8B | Qwen3-14B |
|------|--------|------|-----------|------------|-----------|----------|-----------|
| FP16 | -      | -    | 4.8653    | 4.3560     | 5.4971    | 8.5980   | 7.5960    |
| 4bit | RTN    | QEP  | 5.2303    | 4.5432     | 6.9551    | 11.0644  | 12.7807   |
|      |        | LoaQ | 5.3286    | 4.4831     | 6.1800    | 10.7548  | 8.5981    |
|      |        | Ours | 5.2961    | 4.5237     | 7.4465    | 9.3566   | 8.1412    |
|      | GPTQ   | QEP  | 5.0954    | 4.4875     | 6.3459    | 10.9824  | 12.2558   |
|      |        | LoaQ | 5.1399    | 4.4630     | 6.2109    | 9.9321   | 8.2968    |
|      |        | Ours | 5.0495    | 4.4900     | 6.3818    | 9.1233   | 7.9668    |
| 3bit | RTN    | QEP  | 21.0619   | 6.3838     | 25.3924   | 22.5769  | 18.1640   |
|      |        | LoaQ | 8.9140    | 5.4920     | 14.1467   | >1e3     | 12.1034   |
|      |        | Ours | 6.5760    | 5.3979     | 9.8112    | 12.7110  | 10.8723   |
|      | GPTQ   | QEP  | 6.3966    | 5.1518     | 11.0124   | 15.0779  | 14.2997   |
|      |        | LoaQ | 6.8494    | 4.9777     | 9.0706    | 11.7249  | 10.8154   |
|      |        | Ours | 5.8990    | 5.0785     | 8.7971    | 11.3805  | 9.5815    |
| 2bit | RTN    | QEP  | >1e3      | >1e3       | >1e3      | >1e3     | >1e3      |
|      |        | LoaQ | >1e3      | >1e3       | >1e3      | 375.1837 | 831.7296  |
|      |        | Ours | >1e3      | 552.2888   | >1e3      | 312.4608 | 352.2354  |
|      | GPTQ   | QEP  | 101.1521  | 84.3543    | >1e3      | 165.7484 | 199.1968  |
|      |        | LoaQ | 590.9850  | 24.2423    | 217.8416  | 550.3343 | 43.2624   |
|      |        | Ours | 341.3434  | 26.3311    | 87.5296   | 58.8030  | 46.4656   |

expressed as follows:

$$L(\widehat{W}_V, \widehat{W}_O) = \|\widehat{\Omega} + \widehat{R} - (\Omega + R)\|_F^2,$$
  

$$\widehat{\Omega} = \operatorname{Concat}_{h \in [H]} (\widehat{S}^{(h)}(\widehat{X}\widehat{W}_V^{(g)})) \widehat{W}_O,$$
  

$$\Omega = \operatorname{Concat}_{h \in [H]} (S^{(h)}(XW_V^{(g)})) W_O,$$

where Concath∈[H] denotes concatenation along the head dimension. As in the previous case, the minimization becomes straightforward once either <sup>W</sup>c<sup>V</sup> or <sup>W</sup>c<sup>O</sup> is fixed. Further details are provided in Appendix [A.1.3.](#page-13-0)

Up-Down Module. After the self-attention block, most Transformer architectures process the representations through an MLP layer. We quantize the Up–Down projection in this MLP as a submodule. The objective function is expressed as

$$L(\widehat{W}_U, \widehat{W}_D) = \|\widehat{F} + \widehat{R} - (F + R)\|_F^2,$$
$$\widehat{F} = \left(\Phi(\widehat{X}\widehat{W}_G) \odot \widehat{X}\widehat{W}_U\right)\widehat{W}_D,$$
$$F = (\Phi(XW_G) \odot XW_U)W_D,$$

where Φ(·) denotes the activation function, and LLaMA employs the SiLU function [\(Touvron et al.,](#page-10-11)

[2023\)](#page-10-11). This work restricts the optimization variables to W<sup>U</sup> and W<sup>D</sup> to simplify the minimization process in the relaxation Step. However, LPCD can also be applied by approximately solving the minimization problem concerning WG; investigating the effect of this approach is left for future work. The detailed update rules for this Up-Down submodule are provided in Appendix [A.1.3.](#page-13-0)

# 5 Experiments

We conduct experiments to assess the effectiveness of the proposed method. To evaluate how much LPCD enhances the existing approach, we utilize the final weight of QEP or LoaQ and combine LPCD with the current quantization method.

## 5.1 Setting

Baselines and Quantization Methods. In this study, we focus only on the per-channel weight quantization scheme. We employ representative layer-wise PTQ methods of Round-to-nearest (RTN) and GPTQ, which are used in conjunction with error compensation techniques. Some error compensation methods have been reported to enhance existing layer-wise quantization methods.

Table 2: Zero-shot average accuracy (↑) on ARC-E and PIQA for LLaMA and Qwen models across different bit-widths and quantization methods.

| Bits | Method |      | LLaMA2-7B | LLaMA2-13B | LLaMA3-8B | Qwen3-8B | Qwen3-14B |
|------|--------|------|-----------|------------|-----------|----------|-----------|
| FP16 | -      | -    | 0.7682    | 0.7896     | 0.7915    | 0.7930   | 0.8133    |
| 4bit | RTN    | QEP  | 0.7467    | 0.7825     | 0.7613    | 0.6998   | 0.7369    |
|      |        | LoaQ | 0.7471    | 0.7812     | 0.7812    | 0.6333   | 0.7760    |
|      |        | Ours | 0.7527    | 0.7793     | 0.7789    | 0.7584   | 0.7949    |
|      | GPTQ   | QEP  | 0.7428    | 0.7741     | 0.7599    | 0.7113   | 0.7695    |
|      |        | LoaQ | 0.7456    | 0.7817     | 0.7559    | 0.6066   | 0.7702    |
|      |        | Ours | 0.7504    | 0.7810     | 0.7001    | 0.7661   | 0.7782    |
| 3bit | RTN    | QEP  | 0.5784    | 0.6757     | 0.5136    | 0.5442   | 0.6164    |
|      |        | LoaQ | 0.4799    | 0.6928     | 0.5282    | 0.3971   | 0.6904    |
|      |        | Ours | 0.5688    | 0.7433     | 0.5373    | 0.6291   | 0.6560    |
|      | GPTQ   | QEP  | 0.6881    | 0.7372     | 0.5427    | 0.5986   | 0.6535    |
|      |        | LoaQ | 0.4556    | 0.7462     | 0.5532    | 0.6363   | 0.6222    |
|      |        | Ours | 0.6602    | 0.7431     | 0.6290    | 0.6493   | 0.6919    |
| 2bit | RTN    | QEP  | 0.3816    | 0.3831     | 0.3904    | 0.3845   | 0.3895    |
|      |        | LoaQ | 0.3758    | 0.3805     | 0.3836    | 0.3917   | 0.4183    |
|      |        | Ours | 0.3756    | 0.3794     | 0.3826    | 0.3864   | 0.3864    |
|      | GPTQ   | QEP  | 0.3931    | 0.3821     | 0.3889    | 0.3972   | 0.4355    |
|      |        | LoaQ | 0.3761    | 0.3840     | 0.3842    | 0.4031   | 0.4253    |
|      |        | Ours | 0.3856    | 0.3817     | 0.3962    | 0.4054   | 0.4354    |

We employ QEP and LoaQ as baselines for our proposed method. We perform quantization in the INT4, INT3, and INT2 settings. We skip quantization for the last 2 layers due to the higher frequency of outliers observed in their activations.

Dataset. GPTQ computes Hessian using the calibration dataset to perform effective quantization. Furthermore, error compensation methods compute activations for the output approximation. Following previous studies, LoaQ uses 128 samples of 2048 tokens each from the C4 dataset. We observe over-fitting to the calibration dataset when we employ 128 samples. We employ 2048 tokens consisting of 256 sequences, randomly sampled from the WikiText-2 dataset, as the calibration dataset.

Models. We evaluate the proposed method and baselines in recent major open-weight LLM, including LLaMA2 [\(Touvron et al.,](#page-10-11) [2023\)](#page-10-11), LLaMA3 [\(Grattafiori et al.,](#page-9-20) [2024\)](#page-9-20), and the dense Qwen3 model families. LLaMA is an open-weight LLM family that is primarily developed by Meta Platforms. We employ LLaMA2-7B, LLaMA2- 13B, and LLaMA3-8B for evaluation. Qwen3 is a powerful open-weight LLM family as well as

LLaMA family. They employ a slightly different architecture from LLaMA such as Q/K RMSNorm. We employ Qwen3-4B, Qwen3-8B, and Qwen3- 14B for evaluation.

Hyper Parameters. We conduct a grid search to determine the optimal propagation strength parameter α for QEP and the optimal sub-layer output approximation strength parameter β for LoaQ. The grid search is performed using smaller models, such as Qwen3-0.5B and LLaMA3.2-1B, after which we applied the resulting optimal parameters to the larger models. Following LoaQ, the range of α is from 0 to 1 with increments of 0.1 and, the range of β is from 0 to 1 with increments 0.05. We apply LPCD on the relaxed weight of LoaQ; performances of LoaQ are generally better than those of QEP. We first apply LoaQ to each submodule and then perform LPCD. As explained in Sec. [4.4,](#page-5-1) we apply LPCD to three groups of Transformer submodules: the Q/K module, the V/O module, and the Up/Down module. For gradient-based optimization of LPCD, We employ 8 batch size, 40 epochs, and cosine scheduled learning rate that begin with 10−<sup>5</sup> . The optimization is conducted using the Adam optimizer with the default settings

in PyTorch.

Evaluations. We follow the established evaluation protocols for quantization algorithms used in numerous previous studies. We evaluate the perplexity (PPL) on the WikiText2 dataset. We also evaluated zero-shot accuracy on the ARC Easy and PiQA.benchmarks. We implement QEP, LoaQ, and LPCD using Python 3.12.11 with PyTorch 2.4.0 and Hugging Face Transformers 4.55.3. All experiments were conducted on an NVIDIA H100 GPU using the TSUBAME 4.0 supercomputer.

## 5.2 Result

Perplexities. Table [1](#page-6-0) summarizes the perplexities of various PTQ configurations within the LLaMA and Qwen families. Overall, LPCD-based submodule quantization achieves the lowest perplexity in most settings, consistently outperforming both QEP and LoaQ, irrespective of whether RTN or GPTQ is used. The gains are most pronounced in low-bit regimes: for the practically important LLaMA-3-8B and Qwen-3-8B models at 3-bit and 2-bit levels, LPCD substantially reduces PPL compared to both baselines, preventing the severe degradation or divergence observed with QEP and LoaQ. Notably, for Qwen-3-8B, RTN combined with LPCD already surpasses the more sophisticated QEP+GPTQ configuration, indicating that submodule LPCD provides improvements that are largely orthogonal to the choice of the underlying layer-wise quantizer.

Zero Shot Task Evaluation. Table 2 indicates that LPCD achieves the highest or nearly the highest zero-shot accuracy across various models, bitwidths, and base quantizers. At 4-bits, our method closely matches FP16 performance while slightly improving both QEP and LoaQ, indicating that optimization at the submodule-level does not adversely affect high-precision behavior. The advantages are more apparent in low-bit regimes. For the practically important LLaMA-3-8B and Qwen-3-8B models at 3-bit and 2-bit, LPCD consistently recovers a substantial portion of the accuracy lost by QEP and LoaQ when used with RTN and GPTQ. Remarkably, the simple RTN+LPCD configuration on Qwen-3-8B outperforms the more sophisticated QEP+GPTQ baseline, demonstrating that our submodule refinement complements rather than merely imitates existing layer-wise PTQ techniques.

# 6 Conclusion

We propose LPCD, a unified framework that extends PTQ beyond traditional layer-wise formulations. LPCD optimizes relaxed objectives across arbitrary Transformer submodules and subsequently projects the solutions back using existing layerwise PTQ projectors. In this formulation, GPTQstyle activation-aware PTQ, QEP, and LoaQ are presented as special cases, each corresponding to specific choices of submodules, loss functions, and single-step updates. Thus, LPCD can be viewed as a strict generalization that unifies previously separate approaches into a submodule-centric perspective while preserving the efficiency and modularity of standard layer-wise pipelines.

For the LLaMA and Qwen models, LPCD-based submodule quantization results in lower perplexity and shows competitive or superior zero-shot accuracy compared to QEP and LoaQ across various bitwidths, particularly at 3 and 2 bits, without altering the underlying layer-wise quantizers or the inference stack. These results indicate that optimizing at the submodule level with LPCD yields consistent gains in addition to existing QEP and LoaQ-style refinements, rather than replacing them. Future work includes applying LPCD to more complex nonlinear submodules, jointly handling weights, activations, and KV caches at scale, and integrating LPCD with quantization-aware finetuning to further enhance the deployability of low-bit LLMs.

# Acknowledgements

The authors would like to express their sincere gratitude to Koichi Shirahata and Yuhei Umeda of Fujitsu Limited, and to Katsuki Fujisawa, Toshio Endo, and Yoshihiko Fujisawa of Institute of Science Tokyo for their valuable support and insightful advice. This work was partially supported by JST BOOST, Japan (Grant No. JPMJBY24D0), and by the Cabinet Office, Government of Japan, through the SIP program "Promotion of the Application of Advanced Quantum Technology Platforms to Social Issues".

# References

<span id="page-8-0"></span>Yamato Arai and Yuma Ichikawa. 2025. [Quantization](https://openreview.net/forum?id=a3l3K9khbL) [error propagation: Revisiting layer-wise post-training](https://openreview.net/forum?id=a3l3K9khbL) [quantization.](https://openreview.net/forum?id=a3l3K9khbL) In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*.

<span id="page-8-1"></span>Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian

- Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2024. Quarot: Outlier-free 4-bit inference in rotated llms. *Advances in Neural Information Processing Systems*, 37:100213–100240.
- <span id="page-9-15"></span>Kayhan Behdin, Ayan Acharya, Sathiya Keerthi Aman Gupta, and Rahul Mazumder. 2023. Quantease: Optimization-based quantization for language models-an efficient and intuitive algorithm. *stat*, 1050:5.
- <span id="page-9-11"></span>Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv preprint arXiv:1308.3432*.
- <span id="page-9-7"></span>Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. 2023. Quip: 2-bit quantization of large language models with guarantees. *Advances in Neural Information Processing Systems*, 36:4396– 4429.
- <span id="page-9-0"></span>Lingjiao Chen, Matei Zaharia, and James Zou. 2023. Frugalgpt: How to use large language models while reducing cost and improving performance. *arXiv preprint arXiv:2305.05176*.
- <span id="page-9-3"></span>Hongrong Cheng, Miao Zhang, and Javen Qinfeng Shi. 2024. A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
- <span id="page-9-18"></span>Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. *Advances in neural information processing systems*, 35:30318– 30332.
- <span id="page-9-21"></span>Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023a. Qlora: Efficient finetuning of quantized llms. *Advances in neural information processing systems*, 36:10088–10115.
- <span id="page-9-19"></span>Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. 2023b. Spqr: A sparse-quantized representation for near-lossless llm weight compression. *arXiv preprint arXiv:2306.03078*.
- <span id="page-9-12"></span>Tim Dettmers and Luke Zettlemoyer. 2023. The case for 4-bit precision: k-bit inference scaling laws. In *International Conference on Machine Learning*, pages 7750–7774. PMLR.
- <span id="page-9-13"></span>Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. 2024. Extreme compression of large language models via additive quantization. *arXiv preprint arXiv:2401.06118*.
- <span id="page-9-5"></span>Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. Gptq: Accurate post-training quantization for generative pre-trained transformers. *arXiv preprint arXiv:2210.17323*.

- <span id="page-9-2"></span>Ruihao Gong, Yifu Ding, Zining Wang, Chengtao Lv, Xingyu Zheng, Jinyang Du, Haotong Qin, Jinyang Guo, Michele Magno, and Xianglong Liu. 2024. A survey of low-bit large language models: Basics, systems, and algorithms. *arXiv preprint arXiv:2409.16694*.
- <span id="page-9-20"></span>Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*.
- <span id="page-9-8"></span>Ziyi Guan, Hantao Huang, Yupeng Su, Hong Huang, Ngai Wong, and Hao Yu. 2024. Aptq: Attentionaware post-training mixed-precision quantization for large language models. In *Proceedings of the 61st ACM/IEEE Design Automation Conference*, pages 1–6.
- <span id="page-9-4"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3.
- <span id="page-9-1"></span>Jiedong Lang, Zhehao Guo, and Shuyu Huang. 2024. A comprehensive study on quantization techniques for large language models. In *2024 4th International Conference on Artificial Intelligence, Robotics, and Communication (ICAIRC)*, pages 224–231. IEEE.
- <span id="page-9-9"></span>Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao, and Priyadarshini Panda. 2025. [GPTAQ: Efficient](https://openreview.net/forum?id=QdELyl0FST) [finetuning-free quantization for asymmetric calibra](https://openreview.net/forum?id=QdELyl0FST)[tion.](https://openreview.net/forum?id=QdELyl0FST) In *Forty-second International Conference on Machine Learning*.
- <span id="page-9-6"></span>Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. Awq: Activation-aware weight quantization for ondevice llm compression and acceleration. *Proceedings of Machine Learning and Systems*, 6:87–100.
- <span id="page-9-10"></span>Li Lin and Xiaojun Wan. 2025. Loaq: Layer-wise output approximation quantization. *arXiv preprint arXiv:2509.06297*.
- <span id="page-9-16"></span>Yifei Liu, Jicheng Wen, Yang Wang, Shengyu Ye, Li Lyna Zhang, Ting Cao, Cheng Li, and Mao Yang. 2024a. Vptq: Extreme low-bit vector post-training quantization for large language models. *arXiv preprint arXiv:2409.17066*.
- <span id="page-9-14"></span>Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. 2023. Llm-qat: Data-free quantization aware training for large language models. *arXiv preprint arXiv:2305.17888*.
- <span id="page-9-17"></span>Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. 2024b. Spinquant: Llm quantization with learned rotations. *arXiv preprint arXiv:2405.16406*.

- <span id="page-10-4"></span>Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Burlachenko, Kai Yi, Dan Alistarh, and Peter Richtarik. 2024. Pv-tuning: Beyond straight-through estimation for extreme llm compression. *Advances in Neural Information Processing Systems*, 37:5074–5121.
- <span id="page-10-9"></span>Yuzhang Shang, Zhihang Yuan, Qiang Wu, and Zhen Dong. 2023. Pb-llm: Partially binarized large language models. *arXiv preprint arXiv:2310.00034*.
- <span id="page-10-11"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.
- <span id="page-10-5"></span>Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. 2024. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. *arXiv preprint arXiv:2402.04396*.
- <span id="page-10-7"></span>Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. 2023. Bitnet: Scaling 1-bit transformers for large language models. *arXiv preprint arXiv:2310.11453*.
- <span id="page-10-0"></span>Wenxiao Wang, Wei Chen, Yicong Luo, Yongliu Long, Zhengkai Lin, Liye Zhang, Binbin Lin, Deng Cai, and Xiaofei He. 2024. Model compression and efficient inference for large language models: A survey. *arXiv preprint arXiv:2402.09748*.
- <span id="page-10-2"></span>Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, and Tianyi Zhou. 2024a. A survey on knowledge distillation of large language models. *arXiv preprint arXiv:2402.13116*.
- <span id="page-10-6"></span>Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, and Wanxiang Che. 2024b. Onebit: Towards extremely low-bit large language models. *arXiv preprint arXiv:2402.11295*.
- <span id="page-10-1"></span>Menglin Yang, Jialin Chen, Yifei Zhang, Jiahong Liu, Jiasheng Zhang, Qiyao Ma, Harshit Verma, Qianru Zhang, Min Zhou, Irwin King, and 1 others. 2024. Low-rank adaptation for foundation models: A comprehensive review. *arXiv preprint arXiv:2501.00365*.
- <span id="page-10-3"></span>Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. 2022. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. *Advances in Neural Information Processing Systems*, 35:27168– 27183.
- <span id="page-10-10"></span>Shihao Zhang, Haoyu Zhang, Ian Colbert, and Rayan Saab. 2025. Qronos: Correcting the past by shaping the future... in post-training quantization. *arXiv preprint arXiv:2505.11695*.

<span id="page-10-8"></span>Jiaqi Zhao, Ming Wang, Miao Zhang, Yuzhang Shang, Xuebo Liu, Yaowei Wang, Min Zhang, and Liqiang Nie. 2025. Benchmarking post-training quantization in llms: Comprehensive taxonomy, unified evaluation, and comparative analysis. *arXiv preprint arXiv:2502.13178*.

### **A** Derivation

## <span id="page-11-0"></span>A.1 Update Rule of Submodule PTQ

In this section, we derive the relaxation step for the QK, VO, and Up-Down modules introduced in Sec. 4.4. In all cases, the objective reduces to a linear least-squares problem once one block is fixed, allowing the relaxed minimizer to have a closed-form expression. However, the associated design matrices are exceptionally large in realistic LLM configurations, rendering the exact closed-form solution memory-inefficient. Therefore, we solve the least-squares problems approximately using gradient-based methods

## <span id="page-11-1"></span>A.1.1 QK Module

We consider a single head of grouped-query attention; the multi-head and grouped cases are derived by applying the same derivation either head-wise or group-wise. Let  $S_M \in \mathbb{R}^{T \times T}$  denote the full-precision causal-masked attention score matrix for this head, i.e.,  $S_M = M \odot S$  for some unmasked score matrix S and a fixed causal mask  $M \in \{0,1\}^{T \times T}$ . Let  $\widehat{W}_Q, \widehat{W}_K \in \mathbb{R}^{D_{\mathrm{model}} \times d_k}$  be the quantized query and key projection matrices, where  $d_k$  denotes the head dimension. The RoPE operator  $\mathcal{R}(\cdot)$  is linear in its argument. Therefore, there exists a fixed block-diagonal matrix  $R_\Theta \in \mathbb{R}^{Td_k \times Td_k}$  such that for any  $Z \in \mathbb{R}^{T \times d_k}$ ,  $\mathrm{vec}(\mathcal{R}(Z)) = R_\Theta \mathrm{vec}(Z)$ . The QK-submodule loss for this head is expressed as

$$L(\widehat{W}_Q, \widehat{W}_K) = \left\| M \odot \left( \mathcal{R}(\widehat{X}\widehat{W}_Q) \mathcal{R}(\widehat{X}\widehat{W}_K)^{\top} - S_M \right) \right\|_F^2. \tag{9}$$

Using  $||A||_F^2 = ||\operatorname{vec}(A)||_2^2$  along with  $\operatorname{vec}(AB^\top) = (B \otimes I_T)\operatorname{vec}(A)$  and  $\operatorname{vec}(\widehat{X}\widehat{W}_Q) = (I_{d_k} \otimes \widehat{X})\operatorname{vec}(\widehat{W}_Q)$ , we can express Eq. (9) as

<span id="page-11-2"></span>
$$L(\widehat{W}_Q, \widehat{W}_K) = \left\| D_M Z_Q \operatorname{vec}(\widehat{W}_Q) - \operatorname{vec}(S_M) \right\|_2^2,$$

where  $D_M = \operatorname{diag}(\operatorname{vec}(M)) \in \mathbb{R}^{T^2 \times T^2}$  and

$$Z_Q = \left( \mathcal{R}(\widehat{X}\widehat{W}_K) \otimes I_T \right) R_{\Theta}(I_{d_k} \otimes \widehat{X}) \in \mathbb{R}^{T^2 \times (D_{\text{model}} d_k)}.$$

An analogous expression is obtained when  $\widehat{W}_Q$  is fixed and L is considered a function of  $\widehat{W}_K$ .

Query Relaxation Step. Fix  $\widehat{W}_K$ . Then  $Z_Q$  is constant, and the loss represents a quadratic function of  $u = \text{vec}(\widehat{W}_Q)$ .

$$\operatorname{vec}(\overline{W}_{Q}^{(1)}) = \underset{u}{\operatorname{argmin}} \|D_{M} Z_{Q} u - \operatorname{vec}(S_{M})\|_{2}^{2}.$$
(10)

This is a linear least-squares problem. When  $Z_Q^\top D_M^\top D_M Z_Q$  is invertible, the unique minimizer is

$$\operatorname{vec}(\overline{W}_Q^{(1)}) = \left(Z_Q^{\top} D_M^{\top} D_M Z_Q\right)^{-1} Z_Q^{\top} D_M^{\top} \operatorname{vec}(S_M),$$

and, more generally,

<span id="page-11-3"></span>
$$\operatorname{vec}(\overline{W}_Q^{(1)}) = (D_M Z_Q)^{\dagger} \operatorname{vec}(S_M),$$

where  $(\cdot)^{\dagger}$  denotes the Moore–Penrose pseudoinverse.

**Key Relaxation Step.** Fix  $\widehat{W}_Q$ . By symmetry, we define

$$Z_K = \left( \mathcal{R}(\widehat{X}\widehat{W}_Q) \otimes I_T \right) R_{\Theta} \left( I_{d_k} \otimes \widehat{X} \right) \in \mathbb{R}^{T^2 \times (D_{\text{model}} d_k)}$$

so that

$$L(\widehat{W}_Q, \widehat{W}_K) = \left\| D_M Z_K \operatorname{vec}(\widehat{W}_K) - \operatorname{vec}(S_M) \right\|_2^2.$$

The key relaxation step is therefore:

$$\operatorname{vec}(\overline{W}_K^{(1)}) = \operatorname{argmin} \|D_M Z_K u - \operatorname{vec}(S_M)\|_2^2, \tag{11}$$

which is again a linear least-squares problem. When  $Z_K^\top D_M^\top D_M Z_K$  is invertible, a unique minimizer exists.

$$\operatorname{vec}(\overline{W}_K^{(1)}) = \left(Z_K^{\top} D_M^{\top} D_M Z_K\right)^{-1} Z_K^{\top} D_M^{\top} \operatorname{vec}(S_M),$$

In general,

<span id="page-12-0"></span>
$$\operatorname{vec}(\overline{W}_K^{(1)}) = (D_M Z_K)^{\dagger} \operatorname{vec}(S_M).$$

**Memory Cost.** For each head, the design matrices  $Z_Q$  and  $Z_K$  are of size  $T^2 \times (D_{\text{model}} d_k)$ . Storing  $Z_Q$  or  $Z_K$  explicitly requires  $\Theta(T^2D_{\text{model}}d_k)$  memory, while forming the normal matrices  $Z_Q^\top D_M^\top D_M Z_Q$  and  $Z_K^\top D_M^\top D_M Z_K$  requires  $\Theta(T^2(D_{\text{model}}d_k)^2)$  operations. For typical LLM configurations, this is prohibitive. In our implementation, we therefore do not employ  $Z_Q$  or  $Z_K$  and do not compute the closed-form solutions mentioned above; instead, we approximate the minimizers of Eq. (10) and Eq. (11) using gradient-based least-squares solvers applied directly to the original objective, Eq. (9).

### A.1.2 VO Module

We consider the VO submodule. Let

$$Y = \Omega + R, \quad \widehat{Y} = \widehat{\Omega} + \widehat{R},$$

where R and  $\widehat{R}$  denote the full-precision and quantized residual streams, respectively, and

$$\Omega = \operatorname{Concat}_{h \in [H]} \left( S^{(h)} V^{(g(h))} \right) W_O, \ \ \widehat{\Omega} = \operatorname{Concat}_{h \in [H]} \left( \widehat{S}^{(h)} \widehat{V}^{(g(h))} \right) \widehat{W}_O.$$

Here, g(h) is the group index of head h;  $V^{(g)} = XW_V^{(g)}$  and  $\widehat{V}^{(g)} = \widehat{X}\widehat{W}_V^{(g)}$  are the value projections for group g, and  $S^{(h)}$  and  $\widehat{S}^{(h)}$  are the full-precision and quantized attention weights for head h. The VO submodule loss can be expressed as

$$L(\widehat{W}_V, \widehat{W}_O) = \|\widehat{Y} - Y\|_F^2$$

Value Relaxation Step. Fix  $\widehat{W}_O$  and all other blocks. For a fixed group g, only the value weights  $\widehat{W}_V^{(g)}$  are updated, while all other quantities are treated as constants. Let  $\mathcal{H}_g$  be the set of heads in group g and let  $d_v$  be the per-head value dimension. We decompose the quantized output as

$$\widehat{Y} = \widehat{Y}_{\neg g} + \widehat{Y}_{g}, \quad \widehat{Y}_{g} = \sum_{h \in \mathcal{H}_{g}} \widehat{S}^{(h)} \widehat{X} \widehat{W}_{V}^{(g)} \widehat{W}_{O}^{(h)},$$

where  $\widehat{Y}_{\neg g}$  collects all contributions that do not involve  $\widehat{W}_{V}^{(g)}$ , and  $\widehat{W}_{O}^{(h)} \in \mathbb{R}^{d_v \times D_{\text{model}}}$  represents the output projection for head h. The blockwise objective can be written as

$$\min_{\widehat{W}_{\mathbf{Y}}^{(g)}} \left\| Y - \widehat{Y}_{\neg g} - \widehat{Y}_{g} \right\|_{F}^{2}.$$

By stacking the heads and applying standard vectorization identities, this objective is equivalently formulated as follows:

$$\min_{u} \left\| Z_V^{(g)} u - y^* \right\|_2^2, \quad u = \operatorname{vec}\left(\widehat{W}_V^{(g)}\right), \quad y^* = \operatorname{vec}\left(Y - \widehat{Y}_{\neg g}\right),$$

where

$$Z_V^{(g)} = \sum_{h \in \mathcal{H}_g} \left( \widehat{W}_O^{(h)\top} \otimes \widehat{S}^{(h)} \widehat{X} \right) \in \mathbb{R}^{TD_{\text{model}} \times D_{\text{model}} d_v}.$$

Thus, the relaxed minimizer for  $\widehat{W}_{V}^{(g)}$  is the least-squares solution

<span id="page-13-1"></span>
$$\operatorname{vec}(\overline{W}_{V}^{g,(1)}) = \underset{u}{\operatorname{argmin}} \left\| Z_{V}^{(g)} u - y^{*} \right\|_{2}^{2}.$$

If  $\left(Z_V^{(g)}\right)^{\top}Z_V^{(g)}$  is invertible, the unique minimizer is

$$\operatorname{vec}(\overline{W}_{V}^{g,(1)}) = \left(\left(Z_{V}^{(g)}\right)^{\top} Z_{V}^{(g)}\right)^{-1} \left(Z_{V}^{(g)}\right)^{\top} y^{*},\tag{12}$$

We may replace the inverse with the Moore-Penrose pseudoinverse.

Output-Projection Relaxation Step. Fix  $\widehat{W}_V$  and all other blocks. Let

$$\widehat{H} = \operatorname{Concat}_{h \in [H]} \left( \widehat{S}^{(h)} \widehat{V}^{(g(h))} \right) \in \mathbb{R}^{T \times (Hd_v)},$$

where  $\widehat{V}^{(g(h))} = \widehat{X}\widehat{W}_V^{(g(h))}$  represents the value projection for the group that includes head h. Then, the attention output can be expressed as  $\widehat{\Omega} = \widehat{H}\widehat{W}_O$ , and the blockwise objective for output projection is

<span id="page-13-2"></span>
$$\min_{\widehat{W}_O} \left\| \widehat{H}\widehat{W}_O - (Y - \widehat{R}) \right\|_F^2.$$

This is a standard least-squares matrix problem. If  $\widehat{H}^{\top}\widehat{H}$  is invertible, the relaxed minimizer is expressed in closed form as follows:

$$\overline{W}_O^{(1)} = \left(\widehat{H}^\top \widehat{H}\right)^{-1} \widehat{H}^\top (Y - \widehat{R}), \tag{13}$$

and, more generally  $\overline{W}_O^* = (\widehat{H}^\top \widehat{H})^\dagger \widehat{H}^\top (Y - \widehat{R})$  when  $\widehat{H}^\top \widehat{H}$  is not invertible.

**Memory Cost.** For each group g, the design matrix  $Z_V^{(g)}$  is of size  $TD_{\mathrm{model}} \times (D_{\mathrm{model}} d_v)$ , making the explicit formation of  $Z_V^{(g)}$  and the computation of  $(Z_V^{(g)})^\top Z_V^{(g)}$  memory- and compute-intensive at LLM scale. In practice, we do not materialize  $Z_V^{(g)}$ ; instead, we approximately solve the least-squares problem in Eq. (12) directly using a gradient-based method on the original VO loss. In contrast, the output-projection relaxation step involves only  $\widehat{H} \in \mathbb{R}^{T \times (Hd_v)}$  and  $\widehat{H}^\top \widehat{H} \in \mathbb{R}^{(Hd_v) \times (Hd_v)}$ , whose sizes depend on the number of heads and the value dimension per-head, but not on  $D_{\mathrm{model}}$ . Thus, the closed-form solution in Eq. (13) is computationally feasible, allowing us to compute the O-step exactly without relying on gradient-based optimization.

### <span id="page-13-0"></span>A.1.3 Up-Down Module

Finally, we consider the Up-Down MLP submodule. Let

$$\Phi = \Phi(XW_G), \quad \widehat{\Phi} = \Phi(\widehat{X}\widehat{W}_G), \quad F = (\Phi \odot XW_U)W_D, \quad \widehat{F} = \left(\widehat{\Phi} \odot \widehat{X}\widehat{W}_U\right)\widehat{W}_D,$$

where  $\odot$  denotes the element-wise multiplication. The submodule loss is

$$L(\widehat{W}_U, \widehat{W}_D) = \left\| \widehat{F} + \widehat{R} - (F + R) \right\|_F^2 = \left\| \widehat{F} - Y_{\text{MLP}} \right\|_F^2,$$

with  $Y_{\text{MLP}} = F + R - \hat{R}$ .

Up Relaxation Step. Fix  $\widehat{W}_D$  and all other blocks, and set

$$\widehat{U} = \widehat{X} \widehat{W}_U \in \mathbb{R}^{T \times D_{\mathrm{up}}}, \ \widehat{\Phi} = \Phi(\widehat{X} \widehat{W}_G) \in \mathbb{R}^{T \times D_{\mathrm{up}}}, \ \widehat{F} = \left(\widehat{\Phi} \odot \widehat{U}\right) \widehat{W}_D.$$

Let  $D_{\Phi} = \operatorname{diag}(\operatorname{vec}(\widehat{\Phi}))$ . Using standard vectorization identities, we obtain

$$\operatorname{vec}(\widehat{F}) = Z_U \operatorname{vec}(\widehat{W}_U), \quad Z_U = \left(\widehat{W}_D^{\top} \otimes I_T\right) D_{\Phi}\left(I_{D_{\operatorname{up}}} \otimes \widehat{X}\right) \in \mathbb{R}^{TD_{\operatorname{model}} \times D_{\operatorname{model}} D_{\operatorname{up}}}.$$

Thus, the relaxation step for  $\widehat{W}_U$  reduces to a linear least-squares problem

<span id="page-14-1"></span>
$$\operatorname{vec}(\overline{W}_{U}^{(1)}) = \operatorname{argmin} \|Z_{U}u - \operatorname{vec}(Y_{\operatorname{MLP}})\|_{2}^{2}.$$
(14)

In principle, when  $Z_U^{\top} Z_U$  is invertible, the unique minimizer is given by

$$\operatorname{vec}(\overline{W}_{U}^{(1)}) = (Z_{U}^{\top} Z_{U})^{-1} Z_{U}^{\top} \operatorname{vec}(Y_{\operatorname{MLP}}),$$

or more generally by  $Z_U^{\dagger} \operatorname{vec}(Y_{\operatorname{MLP}})$  using the Moore–Penrose pseudoinverse. However,  $Z_U$  has a size of  $TD_{\operatorname{model}} \times (D_{\operatorname{model}}D_{\operatorname{up}})$  and is prohibitively large to be formed explicitly for LLM-scale models. In practice, we approximate the solution of Eq. (14) using a gradient-based least-squares solver directly on the loss  $\|\widehat{F} - Y_{\operatorname{MLP}}\|_F^2$ , without explicitly constructing  $Z_U$ .

**Down Relaxation Step.** Fix  $\widehat{W}_U$  and all other blocks. Recall that

$$\widehat{F} = \left(\widehat{\Phi} \odot \widehat{X} \widehat{W}_U\right) \widehat{W}_D.$$

Define

$$\widehat{Z}_D := \widehat{\Phi} \odot \widehat{X} \widehat{W}_U \in \mathbb{R}^{T \times D_{\text{up}}}, \ \widehat{F} = \widehat{Z}_D \widehat{W}_D.$$

Substituting this into the submodule loss yields the blockwise objective

$$\min_{\widehat{W}_D} \left\| \widehat{Z}_D \widehat{W}_D - Y_{\text{MLP}} \right\|_F^2.$$

This is a standard matrix least-squares problem concerning  $\widehat{W}_D \in \mathbb{R}^{D_{\text{up}} \times D_{\text{model}}}$ . If  $\widehat{Z}_D^{\top} \widehat{Z}_D$  is invertible, then the unique minimizer is

$$\overline{W}_D^{(1)} = \left(\widehat{Z}_D^{\top} \widehat{Z}_D\right)^{-1} \widehat{Z}_D^{\top} Y_{\text{MLP}},$$

and, more generally,  $\overline{W}_D^{(1)} = \widehat{Z}_D^\dagger Y_{\text{MLP}}$  when  $\widehat{Z}_D^\top \widehat{Z}_D$  is not invertible, where  $\widehat{Z}_D^\dagger$  denotes the Moore–Penrose pseudoinverse of  $\widehat{Z}_D$ .

**Memory Cost.** The design matrix  $Z_U$  has a size of  $TD_{\mathrm{model}} \times (D_{\mathrm{model}}D_{\mathrm{up}})$ , and constructing the normal matrix  $Z_U^{\top}Z_U$  requires  $\Theta(TD_{\mathrm{model}}^2D_{\mathrm{up}}^2)$  operations, which is prohibitive at the scale of LLM. We therefore do not materialize  $Z_U$  and instead solve the least-squares problem in Eq. (14) approximately by applying a gradient-based method directly to  $\|\widehat{F} - Y_{\mathrm{MLP}}\|_F^2$ . By contrast, the down step involves a design matrix of size  $T \times D_{\mathrm{up}}$ , so the associated normal matrix is only  $D_{\mathrm{up}} \times D_{\mathrm{up}}$  and can be handled explicitly. This allows the corresponding least-squares problem to be solved in closed form.

### **B** Additional Theoretical Results

### <span id="page-14-0"></span>**B.1 QEP for Activation Quantization**

In this subsection, we develop a QEP-style extension of activation quantization that was not addressed in the original QEP (Arai and Ichikawa, 2025), and we demonstrate that it emerges naturally as a single LPCD update on the activation block.

When both weights and activations are quantized, several design patterns are commonly used in practice. One approach quantizes weights and activations independently, typically through direct approximation. A second approach first quantizes the activations and then optimizes the weights based on an activation-aware objective defined in terms of the quantized activations. A third approach jointly optimizes the weight and activation quantizers to directly minimize output-level discrepancy, often employing one of the previous strategies for initialization. The LPCD offers a natural method to enhance the second and third approaches by deriving a QEP-style update for the activations, analogous to the original QEP update for the weights. Recall the two-block objective used in the weight-side analysis in Section 4.3:

<span id="page-15-3"></span>
$$L(\widehat{W}, \widehat{X}) = \|\widehat{X}\widehat{W} - XW\|_F^2, \tag{15}$$

In the activation-side variant, the weight block  $\widehat{W}$  is treated as fixed, while the LPCD step is applied to the activation block  $\widehat{X}$ .

**Relaxation Step.** The relaxation step can be expressed as

<span id="page-15-1"></span>
$$\overline{X}^{(1)} = \underset{U}{\operatorname{argmin}} \|U\widehat{W} - XW\|_F^2. \tag{16}$$

A standard matrix calculus computation demonstrates that the gradient is

$$\nabla_U \|U\widehat{W} - XW\|_F^2 = 2(U\widehat{W} - XW)\widehat{W}^\top.$$

Any minimizer  $\hat{X}^*$  of Eq. (16) satisfies the first-order optimality condition:

$$(\widehat{X}^*\widehat{W} - XW)\widehat{W}^\top = 0.$$

Assume the matrix  $\widehat{W}\widehat{W}^{\top} \in \mathbb{R}^{N \times N}$  is invertible. In this case, the unique solution of Eq. (16) is  $\widehat{X}^* = XW\widehat{W}^{\top}(\widehat{W}\widehat{W}^{\top})^{-1}$ . If  $\widehat{W}\widehat{W}^{\top}$  is not invertible, the same derivation shows that any minimizer can be expressed as

$$\overline{X}^{(1)} = XW\widehat{W}^{\top}(\widehat{W}\widehat{W}^{\top})^{\dagger} + Z(I_N - \widehat{W}\widehat{W}^{\top}(\widehat{W}\widehat{W}^{\top})^{\dagger}),$$

for some  $Z \in \mathbb{R}^{T \times N}$ , where  $(\cdot)^\dagger$  denotes the Moore–Penrose pseudoinverse. The choice Z=0 yields the minimum-norm least-squares solution  $\overline{\widehat{X}}^{(1)} = XW\widehat{W}^\top (\widehat{W}\widehat{W}^\top)^\dagger$ , which coincides with the solution obtained when  $\widehat{W}\widehat{W}^\top$  is invertible. In either case, the continuous LPCD relaxation produces a *corrected* activation matrix  $\overline{\widehat{X}}^{(1)}$  that best aligns the quantized layer output  $U\widehat{W}$  with the full-precision output XW in the least-squares sense.

**Projection Step.** The second LPCD step enforces the activation quantization constraint by projecting  $\overline{\widehat{X}}^{(1)}$  onto the set  $\mathbb{Q}^{T\times N}$  using the direct projector  $\Pi^{(d)}_{\mathbb{Q}}$ . Applying this projection to  $\widehat{X}^*$  results in the activation-side QEP update

<span id="page-15-2"></span>
$$\widehat{X}^{\text{QEP}} = \Pi_{\mathbb{Q}}^{(d)}(\overline{X}^{(1)}) = \Pi_{\mathbb{Q}}^{(d)} \left( X W \widehat{W}^{\top} (\widehat{W} \widehat{W}^{\top})^{-1} \right). \tag{17}$$

Eq. (16)–(17) demonstrates that extending QEP from weights to activations involves applying a single LPCD update to the activation block (the block with index r=2 in the two-block objective Eq. (15)), while the weight block  $\widehat{W}$  remains fixed. This construction yields a mathematically consistent activation analog of QEP and integrates both the weight-side and activation-side QEP within the unified LPCD.

### <span id="page-15-0"></span>**B.2 QEP for KV-Cache Quantization**

This section extends QEP and LPCD to KV-cache quantization. We demonstrate that both the key and value caches allow for QEP-style corrections, which can be interpreted as single LPCD updates on their respective activation blocks, followed by standard activation quantizers.

**Setting.** We consider a single Transformer layer and a single attention head for clarity; the multi-head case follows by applying the same derivations to each head independently. Let

$$Q \in \mathbb{R}^{T_q \times d_k}$$
,  $K \in \mathbb{R}^{T_k \times d_k}$ ,  $V \in \mathbb{R}^{T_k \times d_v}$ 

denote the full-precision query, key, and value matrices, respectively. To simplify, we ignore the scaling factor and causal masking. These can be absorbed into Q and K, as shown in Appendix A.1.1. the full-precision attention logits and outputs are

$$S = QK^{\top} \in \mathbb{R}^{T_q \times T_k}, \ A = \text{Softmax}(S), \ Y = AV \in \mathbb{R}^{T_q \times d_v},$$

where the softmax is applied row-wise. At inference time, we store a quantized KV cache  $(\widehat{K}, \widehat{V})$  and use quantized queries  $\widehat{Q}$ . Our goal is to adjust  $\widehat{K}$  and  $\widehat{V}$  within LPCD while preserving a standard activation quantization interface.

## **B.2.1** Kye-Cache Update

We derive a QEP-style update for the key cache. Following the QEP design principle, we align the *pre-softmax* attention scores computed from the quantized path with their full-precision counterparts. We therefore minimize the Frobenius norm

<span id="page-16-0"></span>
$$L_K(\widehat{K}) = \|\widehat{Q}\widehat{K}^\top - QK^\top\|_F^2,\tag{18}$$

where  $\widehat{Q}$  denotes the quantized query matrix produced by the preceding submodule. This corresponds to the block-wise objective of LPCD with the key block treated as an activation block.

**Relaxation Step.** Let  $U = \hat{K}^{\top} \in \mathbb{R}^{d_k \times T_k}$  represent the relaxed variable. The objective Eq. (18) is defined as

$$L_K(U) = \|\widehat{Q}U - S\|_F^2, \ S = QK^\top.$$

The optimality condition for this least-squares problem is as follows:

$$\widehat{Q}^{\top}(\widehat{Q}U - S) = 0.$$

Defining  $\hat{H}_Q = \hat{Q}^{\top} \hat{Q}$  and  $C_Q = \hat{Q}^{\top} (Q - \hat{Q})$ . We obtain

$$\widehat{H}_Q U = \widehat{Q}^\top S = \widehat{Q}^\top Q K^\top = (\widehat{H}_Q + C_Q) K^\top.$$

Assuming that  $\widehat{H}_Q$  is invertible or replacing it with a regularized inverse, the unique minimizer is

$$U^* = \widehat{H}_O^{-1}(\widehat{H}_Q + C_Q)K^{\top} = (I + \widehat{H}_O^{-1}C_Q)K^{\top}.$$

Transposing back to the original parameterization yields the relaxed key cache

$$\overline{K}^{(1)} = (U^*)^{\top} = K \Big( I + \hat{H}_Q^{-1} C_Q \Big)^{\top}.$$

As in QEP for weights, we introduce a tunable coefficient  $\alpha_K \in [0, 1]$  to interpolate between the original and fully corrected keys:

$$\overline{K}^{(1)}(\alpha_K) = K \left( I + \alpha_K \widehat{H}_Q^{-1} C_Q \right)^{\top}. \tag{19}$$

When  $\alpha_K = 0$ , we obtain  $\overline{K}^{(1)} = K$ ; when  $\alpha_K = 1$ , we recover the full least-squares solution.

**Projection Step.** Within our LPCD formulation, the key cache is treated as an activation block, i.e.,  $r \in \mathcal{R}_a$ . Thus, the projection step applies the direct activation quantizer  $\Pi^{(d)}_{\mathbb{Q}}$  to the relaxed keys:

<span id="page-16-2"></span><span id="page-16-1"></span>
$$\widehat{K}^{(1)} = \Pi_{\mathbb{Q}}^{(d)} \left( \overline{K}^{(1)}(\alpha_K) \right), \tag{20}$$

where  $\Pi_{\mathbb{Q}}^{(d)}$  is instantiated using either per-channel or per-token schemes (e.g., KIVI-style asymmetric quantization). Eq. (19)–(20) thus defines a single LPCD update for the key-cache block with a QEP-style correction.

### **B.2.2** Value-Cache Update

Next, we derive a QEP-style update for the value cache. Here, we align the *post-softmax* attention outputs by using the quantized attention weights as the design matrix. Let  $\hat{S} = \hat{Q}\hat{K}^{\top}$  and  $\hat{A} = \operatorname{Softmax}(\hat{S})$  be the (fixed) logits and attention weights computed from the quantized queries and keys. We define the loss

$$L_V(\widehat{V}) = \left\| \widehat{A}\widehat{V} - AV \right\|_F^2 = \left\| \widehat{A}\widehat{V} - Y \right\|_F^2, \tag{21}$$

which again matches the QEP pattern.

**Relaxation Step.** Let  $U = \hat{V} \in \mathbb{R}^{T_k \times d_v}$  denote the relaxed variable. The objective expressed in Eq. (21) can be stated as

$$L_V(U) = \|\widehat{A}U - Y\|_F^2, \ Y = AV.$$

The normal equations are:

<span id="page-17-1"></span>
$$\widehat{A}^{\top} \Big( \widehat{A} U - Y \Big) = 0.$$

By defining  $\widehat{H}_A = \widehat{A}^{\top} \widehat{A}$  and  $C_A = \widehat{A}^{\top} (A - \widehat{A})$ , we obtain

$$\widehat{H}_A U = \widehat{A}^\top Y = \widehat{A}^\top A V = (\widehat{H}_A + C_A) V.$$

Assuming  $\widehat{H}_A$  is invertible, the least-squares minimizer is

$$U^* = \hat{H}_A^{-1}(\hat{H}_A + C_A)V = (I + \hat{H}_A^{-1}C_A)V.$$

Thus, the relaxed value cache is

$$\overline{V}^{(1)} = \left(I + \widehat{H}_A^{-1} C_A\right) V.$$

Introducing a propagation strength parameter  $\alpha_V \in [0,1]$  yields

$$\overline{V}^{(1)}(\alpha_V) = \left(I + \alpha_V \widehat{H}_A^{-1} C_A\right) V,$$

which interpolates between the original values and the complete solution.

**Projection Step.** Regarding the key cache, the value cache is considered an activation block in our LPCD. We therefore apply the direct activation projection:

<span id="page-17-2"></span>
$$\widehat{V}^{(1)} = \Pi_{\mathbb{Q}}^{(d)} \left( \overline{V}^{(1)}(\alpha_V) \right). \tag{22}$$

In practice, per-token schemes are often preferable for values, whereas keys benefit from per-channel schemes; our formulation is agnostic to this choice and simply reuses the underlying activation quantizer.

Eqs. (18)–(22) demonstrate that the QEP for KV caches is derived by executing a single LPCD update on the key and value-cache blocks with suitably chosen block-wise objectives. The key update aligns pre-softmax attention logits, while the value update aligns post-softmax attention outputs. Both updates are expressed in closed form during the relaxation step and then projected using standard activation quantizers, thereby preserving compatibility with existing KV-cache quantization schemes, such as perchannel and per-token uniform quantization. In Sec. 4.4, we use these KV-cache updates as building blocks for submodule LPCD applied to the grouped QK and VO modules.

#### <span id="page-17-0"></span>**B.3 QEP for Orthogonal Rotation Matrices**

Rotation-based incoherence processing (Frantar et al., 2022; Tseng et al., 2024; Liu et al., 2024b) requires a linear map represented by an orthogonal matrix  $R \in \mathbb{R}^{N \times N}$  to redistribute outliers across channels. For a weight matrix  $W \in \mathbb{R}^{N \times M}$  and activations  $X \in \mathbb{R}^{T \times N}$ , the transformation

<span id="page-17-3"></span>
$$Y = XW = (XR)(R^{\top}W) \tag{23}$$

leaves the full-precision output unchanged while operating in the channel dimension. In rotation-aware PTQ, one typically quantizes the rotated quantities XR and  $R^\top W$ ; their quantized counterparts are referred to as  $\widehat{X} \in \mathbb{R}^{T \times N}$  and  $\widehat{W} \in \mathbb{R}^{N \times M}$ , respectively. In this section, we treat the rotation matrix R as a block in LPCD and derive a single update for R while keeping  $\widehat{X}$  and  $\widehat{W}$  fixed. Compared to optimization methods such as CalySGD introduced in SpinQuant, the QEP-style updates described below are more memory-efficient, as only the linear layers need to be stored in memory. Additionally, it provides more stable optimization than CalySGD.

**Objective.** Motivated by the factorization in Eq. (23), we select R so that the rotated full-precision tensors XR and  $R^{\top}W$  are well aligned with their corresponding fixed quantized counterparts  $\widehat{X}$  and  $\widehat{W}$ . We therefore consider the quadratic objective

$$L_R(R) = \lambda_a ||XR - \widehat{X}||_F^2 + \lambda_w ||R^\top W - \widehat{W}||_F^2,$$
(24)

where  $\lambda_a, \lambda_w \geq 0$  balances the contributions of activation and weight. This defines the LPCD block-loss for the rotation block  $M_r = R$  while keeping all other blocks fixed.

**Relaxation Step.** We relax the orthogonality constraint and minimize Eq. (24) over all real matrices  $R \in \mathbb{R}^{N \times N}$ . The first term in Eq. (24) can be expressed as

$$f_a(R) = ||XR - \widehat{X}||_F^2 = \operatorname{tr}\left[(XR - \widehat{X})^\top (XR - \widehat{X})\right].$$

Its gradient is

<span id="page-18-0"></span>
$$\nabla_R f_a(R) = 2X^{\top} (XR - \widehat{X}).$$

For the second term,

$$f_w(R) = ||R^\top W - \widehat{W}||_F^2 = ||W^\top R - \widehat{W}^\top||_F^2,$$

we obtain

$$\nabla_R f_w(R) = 2W(W^\top R - \widehat{W}^\top) = 2(WW^\top R - W\widehat{W}^\top).$$

Combining both contributions, the total gradient of  $L_R(R)$  is expressed as

$$\nabla_R L_R(R) = 2\lambda_a X^{\top} (XR - \widehat{X}) + 2\lambda_w (WW^{\top}R - W\widehat{W}^{\top}).$$

Setting the gradient to zero yields the normal equations

$$(\lambda_a X^\top X + \lambda_w W W^\top) R = \lambda_a X^\top \widehat{X} + \lambda_w W \widehat{W}^\top.$$

We define  $H_R = \lambda_a X^\top X + \lambda_w W W^\top$  and  $B_R = \lambda_a X^\top \widehat{X} + \lambda_w W \widehat{W}^\top$ . Assuming  $H_R$  is invertible or that a regularized inverse is used, the relaxed minimizer can be expressed in closed form as follows:

<span id="page-18-2"></span><span id="page-18-1"></span>
$$\overline{R}^{(1)} = H_R^{-1} B_R. (25)$$

This step corresponds to the LPCD relaxation for the rotation block: we compute the continuous matrix  $\overline{R}^{(1)}$  that satisfies both the activation and weight constraints in the least-squares sense.

**Projection Step.** The rotation matrix is required to be orthogonal,

$$\mathcal{O}(N) = \{ R \in \mathbb{R}^{N \times N} : R^{\top} R = I_N \}.$$

To restore this constraint, we project the relaxed solution  $\overline{R}^{(1)}$  onto  $\mathcal{O}(N)$  using the Frobenius norm:

$$R^{(1)} = \Pi_{\mathcal{O}(N)} \left( \overline{R}^{(1)} \right) := \underset{R \in \mathcal{O}(N)}{\operatorname{argmin}} \left\| R - \overline{R}^{(1)} \right\|_F^2. \tag{26}$$

This is the classical orthogonal Procrustes problem. Let the singular value decomposition of  $\overline{R}^{(1)}$  be

$$\overline{R}^{(1)} = U\Sigma V^{\top},$$

with  $U, V \in \mathbb{R}^{N \times N}$  orthogonal and  $\Sigma$  diagonal matrices that have nonnegative entries. Then the unique minimizer of Eq. (26) is

<span id="page-19-0"></span>
$$R^{(1)} = UV^{\top}. (27)$$

If one wishes to restrict to proper rotations,  $\det(R^{(1)})=1$ ), the sign of the last column of U or V can be flipped accordingly. Eqs. (25) and (27) collectively define a single LPCD update for the rotation block with fixed quantized  $\widehat{X}$  and  $\widehat{W}$ .

### **B.4 QEP for LoRA-Based Error Compensation**

In this section, we extend the LPCD to a setting in which the quantization error is compensated by a low-rank adapter that follows the LoRA framework (Hu et al., 2022; Dettmers et al., 2023a). We treat the LoRA correction as a separate block variable and describe its update solely in terms of the relaxation and projection steps of LPCD.

**LoRA Parameterization.** Suppose we have a *base* quantized weight  $\widehat{W}_0 \in \mathbb{Q}^{N \times M}$  obtained using a layer-wise PTQ method such as GPTQ or AWQ. LoRA parameterizes an additive correction as follows:

$$\widehat{W} = \widehat{W}_0 + \Delta W, \quad \Delta W = BA,$$

where  $B \in \mathbb{R}^{N \times r}$ ,  $A \in \mathbb{R}^{r \times M}$ , and  $r \ll \min\{N, M\}$ . Thus, any LoRA-style model corresponds to a weight matrix within the affine set

$$\widehat{W} \in \widehat{W}_0 + \mathcal{M}_r, \quad \mathcal{M}_r := \{ E \in \mathbb{R}^{N \times M} : \operatorname{rank}(E) \le r \}.$$

We consider the following two-block objective:

$$L(\widehat{W}_0, E) := \left\| \widehat{X}(\widehat{W}_0 + E) - XW \right\|_{E}^{2},$$

with blocks  $M_1 = \widehat{W}_0 \in \mathbb{Q}^{N \times M}$ , the base quantized weight, and  $M_2 = E \in \mathbb{R}^{N \times M}$ , the LoRA correction. We fix  $M_1 = \widehat{W}_0$  and apply one LPCD update to  $M_2$ .

Relaxation Step. Define the output residual of the base quantized weight

$$R \coloneqq XW - \widehat{X}\widehat{W}_0 \in \mathbb{R}^{T \times M},$$

so that

$$L(\widehat{W}_0, E) = \|\widehat{X}E - R\|_F^2.$$

The relaxation step minimizes L over E without imposing any rank constraints:

$$\overline{E}^{(1)} = \underset{U \in \mathbb{R}^{N \times M}}{\operatorname{argmin}} \|\widehat{X}U - R\|_F^2.$$

Let  $\hat{H} \coloneqq \hat{X}^{\top} \hat{X} \in \mathbb{R}^{N \times N}$ . The first-order optimality condition is expressed as follows:

$$\widehat{H}\overline{E}^{(1)} = \widehat{X}^{\top}R.$$

If  $\widehat{H}$  is invertible, the unique minimizer is given by

$$\overline{E}^{(1)} = \widehat{H}^{-1} \widehat{X}^{\top} R = \widehat{H}^{-1} \widehat{X}^{\top} (XW - \widehat{X}\widehat{W}_0).$$

In general, we utilize the Moore–Penrose pseudoinverse and define

$$\overline{E}^{(1)} = \widehat{H}^{\dagger} \widehat{X}^{\top} R,$$

which represents the minimum-norm least-squares solution.

**Projection Step.** The projection step enforces the LoRA rank constraint by projecting  $\overline{E}^{(1)}$  onto the set  $\mathcal{M}_r$  in a manner that aligns with the activation-aware metric. We use the  $\widehat{H}$ -weighted norm

$$||E||_{\widehat{H}}^2 := \operatorname{tr}(E^{\top} \widehat{H} E) = ||\widehat{H}^{1/2} E||_F^2.$$

We define the following projection:

$$E^{(1)} = \Pi_r^{(\text{LoRA})}(\overline{E}^{(1)}) := \underset{E: \text{rank}(E) \le r}{\operatorname{argmin}} \|E - \overline{E}^{(1)}\|_{\widehat{H}}^2.$$

Let  $F^{(1)} \coloneqq \widehat{H}^{1/2}\overline{E}^{(1)} \in \mathbb{R}^{N \times M}$  and compute its singular value decomposition  $F^{(1)} = U\Sigma V^{\top}$ . Denote by  $U_r \in \mathbb{R}^{N \times r}$ ,  $\Sigma_r \in \mathbb{R}^{r \times r}$ , and  $V_r \in \mathbb{R}^{M \times r}$  the matrices obtained by truncating to the top r singular values. The weighted best rank-r approximation of  $\overline{E}^{(1)}$  is given by

$$E^{(1)} = \widehat{H}^{-1/2} U_r \Sigma_r V_r^{\top}.$$

The updated weight after one LPCD iteration on the LoRA block is

$$\widehat{W}^{(1)} = \widehat{W}_0 + E^{(1)}.$$

If an explicit LoRA factorization is required, we define  $B\coloneqq \widehat{H}^{-1/2}U_r\Sigma_r^{1/2}$  and  $A\coloneqq \Sigma_r^{1/2}V_r^{\top}$ , such that  $BA=E^{(1)}$  and  $\widehat{W}^{(1)}=\widehat{W}_0+BA$ . In this manner, error compensation using LoRA is achieved through a single LPCD update on the low-rank correction block, without altering the base layer-wise PTQ operator.