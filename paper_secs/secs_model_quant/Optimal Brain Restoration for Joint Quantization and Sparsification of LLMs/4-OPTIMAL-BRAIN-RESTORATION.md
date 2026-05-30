# 4 OPTIMAL BRAIN RESTORATION

In this work, we propose the Optimal Brain Restoration (OBR) framework, which adjusts weight distributions to reconcile the conflicting demands of pruning and quantization. Following previous practices (Harma et al., 2024; Guo et al., 2024), we adopt an order of pruning-then-quantization. As shown in Fig. 2, the overall process to generate low-bit and sparse weights using the proposed OBR can be formalized as:

$$\hat{\mathbf{W}} = \text{quant}(\text{prune}(\text{rotate}(\mathbf{W})) + \Delta \mathbf{W}^{OBR}), \tag{1}$$

where **W** is the original LLM weights,  $\Delta \mathbf{W}^{OBR}$  is the compensation derived from OBR. In the following, we start in Sec. 4.1 by defining the necessary notations and objective function. Then we detail the generic formulation of the proposed OBR in Sec. 4.2, followed by the specific instantiations for quantization and pruning in Sec. 4.3.

### <span id="page-3-1"></span>4.1 OBJECTIVE APPROXIMATION

Given the weight matrix  $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in}}$  in one standard linear layer and  $\mathbf{X} \in \mathbb{R}^{C_{in} \times L}$  being the input activation representing the dataset's statistics, our work employs the following classic optimization objective (LeCun et al., 1989; Frantar & Alistarh, 2022) which minimizes the perturbation of downstream task loss:

<span id="page-3-2"></span>min 
$$\mathbb{E}[\Delta \mathcal{L}] = \mathbb{E}[\mathcal{L}(\mathbf{X}, \mathbf{W} + \Delta \mathbf{W}) - \mathcal{L}(\mathbf{X}, \mathbf{W})],$$
 (2)

where  $\Delta \mathbf{W}$  is the perturbation on  $\mathbf{W}$ ,  $\mathcal{L}$  is the downstream task loss.

To solve the optimization problem in Eq. (2), we first simplify the objective function. In detail, applying Taylor series on  $\mathcal{L}(\mathbf{X}, \mathbf{W} + \Delta \mathbf{W})$  at  $\mathbf{W}$  drives:

<span id="page-3-3"></span>
$$\Delta \mathcal{L} = \langle \nabla_{\mathbf{W}} \mathcal{L}(\mathbf{X}, \mathbf{W}), \Delta \mathbf{W} \rangle + \frac{1}{2} \text{vec}(\Delta \mathbf{W}) \mathbf{H}_{\text{full}} \text{vec}(\Delta \mathbf{W})^{\top} + \mathcal{O}(\|\Delta \mathbf{W}\|^{3}), \tag{3}$$

where  $\nabla_{\mathbf{W}} \mathcal{L}(\mathbf{X}, \mathbf{W})$  is the gradient,  $\operatorname{vec}(\cdot) : \mathbb{R}^{C_{out} \times C_{in}} \to \mathbb{R}^{1 \times C_{out} C_{in}}$  is the vectorisation operator, and  $\mathbf{H}_{\operatorname{full}} \triangleq \frac{\partial^2 \mathcal{L}}{\partial \operatorname{vec}(\mathbf{W}) \partial \operatorname{vec}(\mathbf{W})^{\top}} \in \mathbb{R}^{C_{out} C_{in} \times C_{out} C_{in}}$  is the layer-wise Hessian.

Assume that the model has been fully trained and reaches a local minima, so the  $\nabla_{\mathbf{W}} \mathcal{L}(\mathbf{X}, \mathbf{W}) \approx 0$ . Further ignoring the last high order terms, Eq. (3) can be approximated into:

$$\Delta \mathcal{L} \approx \frac{1}{2} \text{vec}(\Delta \mathbf{W}) \mathbf{H}_{\text{full}} \text{vec}(\Delta \mathbf{W})^{\top}.$$
 (4)

Despite the above preliminary approximation, computing  $\mathbf{H}_{\text{full}}$  exactly is still infeasible in LLMs due to the  $\mathcal{O}((C_{out}C_{in})^2)$  complexity, we thus following previous works (Frantar & Alistarh, 2022) and estimate  $\mathbf{H}_{\text{full}}$  as:

<span id="page-3-4"></span>
$$\mathbf{H}_{\mathrm{full}} \approx \mathbf{G} \otimes \mathbf{H},$$
 (5)

where  $\mathbf{G} \in \mathbb{R}^{C_{out} \times C_{out}}$  is the output-side curvature matrix which depicts the second-order sensitivity among output channels,  $\mathbf{H} \triangleq 2\mathbf{X}\mathbf{X}^{\top} \in \mathbb{R}^{C_{in} \times C_{in}}$  is the empirical Fisher matrix, and  $\otimes$  denotes the Kronecker product.

Based on Eq. (5), we propose to decouple the row-wise correlation of output channels in  $\mathbf{H}_{\mathrm{full}}$  by approximating  $\mathbf{G}$  as an Identity matrix  $\mathbf{I}$  to make  $\mathbf{H}_{\mathrm{full}} \approx \mathbf{I} \otimes \mathbf{H}$  completely tractable. Finally, the original objective can be simplified into the following  $C_{out}$  independent optimization sub-problems:

<span id="page-4-2"></span>
$$\min \quad \mathbb{E}[\frac{1}{2}\text{vec}(\Delta \mathbf{W})(\mathbf{I} \otimes \mathbf{H})\text{vec}(\Delta \mathbf{W})^{\top}] = \frac{1}{2}\sum_{i=1}^{C_{out}} \mathbb{E}[\Delta \mathbf{w}_i \mathbf{H} \Delta \mathbf{w}_i^{\top}], \tag{6}$$

where  $\Delta \mathbf{w}_i \in \mathbb{R}^{1 \times C_{in}}$  is the *i*-th row of  $\Delta \mathbf{W}$ . Intuitively, Eq. (6) quantifies the impact of weight changes on the final downstream performance. For example, when  $\mathbf{H}$  is large, even a small change in weights can result in large differences for downstream tasks.

#### <span id="page-4-0"></span>4.2 SOLUTION AND FRAMEWORK

To solve the simplified objective in Eq. (6), our proposed OBR employs the Group Error Compensation to optimally adjust weight distributions by shifting information from error-sensitive groups to the other robust ones. Since the rotation matrix acts on both  $\mathbf{W}$  and  $\mathbf{X}$ , and thus cancels out during multiplication, in the following sections, we will omit the rotation operation and directly denote  $\mathbf{W}$  as the rotated matrix for notational clarity.

Let  $\mathcal{J}_i = \frac{1}{2} \Delta \mathbf{w}_i \mathbf{H} \Delta \mathbf{w}_i^{\top}$  denote the *i*-th sub-problem, we begin by partitioning the elements of the *i*-th row  $\Delta \mathbf{w}_i$  into two disjoint groups using two index sets, *i.e.*, the retain set  $R_i$  and the eviction set  $E_i$ , where  $R_i \cup E_i = \{1, \dots, C_{in}\}$  and  $R_i \cap E_i = \emptyset$ . The retain set  $R_i$  collects weights that are less affected by compression, *e.g.*, unpruned or less quantization-distorted, whereas the eviction set  $E_i$  corresponds to the indices of elements that are susceptible to compression effects. For clarity, we will omit the row index *i* in the following.

With this grouping, our key idea is to compensate for compression-induced errors  $\mathbf{e}_E$  in eviction set E by transferring its lost information to a more robust retain set R. To enable this, we reorder the perturbation vector  $\Delta \mathbf{w}$  into  $[\Delta \mathbf{w}_R, \Delta \mathbf{w}_E]$ . Then the sub-problem becomes:

<span id="page-4-3"></span>
$$\underset{\Delta \mathbf{w}_{R}}{\operatorname{arg \, min}} \quad \mathcal{J} = \frac{1}{2} \Delta \mathbf{w} \mathbf{H} \Delta \mathbf{w}^{\top} = \frac{1}{2} [\Delta \mathbf{w}_{R} \quad \mathbf{e}_{E}] \begin{bmatrix} \mathbf{H}_{RR} & \mathbf{H}_{RE} \\ \mathbf{H}_{ER} & \mathbf{H}_{EE} \end{bmatrix} \begin{bmatrix} \Delta \mathbf{w}_{R}^{\top} \\ \mathbf{e}_{E}^{\top} \end{bmatrix}. \tag{7}$$

Since Eq. (7) is an unconstrained optimization problem, we can directly obtain the closed-form solution by taking the partial derivatives w.r.t.  $\Delta \mathbf{w}_R$ , i.e.,  $\nabla_{\Delta \mathbf{w}_R} \mathcal{J} = \mathbf{H}_{RR} \Delta \mathbf{w}_R + \mathbf{H}_{RE} \mathbf{e}_E \triangleq 0$ . Then the optimal solution for  $\Delta \mathbf{w}_R$  which minimizes the row-wise error can be derived as:

<span id="page-4-4"></span>
$$\Delta \mathbf{w}_R^{\star} = -\mathbf{H}_{RR}^{-1} \mathbf{H}_{RE} \mathbf{e}_E. \tag{8}$$

In Fig. 3(a), we give an example on how to extract sub-Hassian  $\mathbf{H}_{RR}$  and  $\mathbf{H}_{RE}$  from  $\mathbf{H}$ . According to the above formulation, the error in set E is theoretically zero guaranteed by the closed-form solution. Since the retain set R is assumed to be robust against compression-related errors, the total error can be decreased through transferring information from E to R. Notably, Eq. (8) also offers a strong explanation that the Hessian actually serves as a "bridge" for error propagation between different groups. Specifically, in Eq. (8), the  $\mathbf{e}_E$  is first projected from E's space to the shared space via  $\mathbf{H}_{RE}$ , followed by the mapping to the R's space through  $\mathbf{H}_{RR}^{-1}$ , and the negative sign denoting the correction direction.

#### <span id="page-4-1"></span>4.3 Specific Implementation

In this section, we apply the generic closed-form solution in Eq. (8) to the specific implementation for sparsification and quantization.

**OBR for Sparsification.** As shown in Fig. 3(b), given the 0-1 mask from existing pruning algorithms, we define retain set  $R_1$  as the unpruned slots, and eviction set  $E_1$  as the pruned ones. In this way, the information loss due to pruning on set  $E_1$  can be compensated by transferring to set  $R_1$ . Formally, since the pruning error on set  $E_1$  is  $\mathbf{e}_{E_1}^{prune} = \mathbf{w}_{E_1}$ , using Eq. (8), the optimal OBR compensation for pruning can be derived as:

$$\Delta \mathbf{w}_{R_1}^{prune} = -\mathbf{H}_{R_1 R_1}^{-1} \mathbf{H}_{R_1 E_1} \mathbf{w}_{E_1}. \tag{9}$$

We then add  $\Delta \mathbf{w}_{R_1}^{prune}$  to the unpruned elements  $\mathbf{w}_{R_1}$  to obtain the OBR-compensated sparse weight  $\bar{\mathbf{w}} = [\mathbf{w}_{R_1} + \Delta \mathbf{w}_{R_1}^{prune}, \mathbf{0}]$ . After that, we perform another round of OBR on  $\bar{\mathbf{w}}$  to further consider the incoming quantization error. Details are given below.

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 3: (a) Given a Hessian approximation  $\mathbf{H}$ , we extract the submatrices  $\mathbf{H}_{RR}$  and  $\mathbf{H}_{RE}$  based on the index sets R and E. (b) The rotated dense weights are partitioned into  $R_1$  and  $E_1$  according to the binary pruning mask, followed by OBR to transfer information from  $\mathbf{w}_{E_1}$  to  $\mathbf{w}_{R_1}$ . (c) The unpruned index set  $R_1$  is further divided into two groups: the first  $\alpha$  fraction assigned to set  $E_2$ , the remaining  $1-\alpha$  to set  $R_2$ . OBR is used to compensate for quantization error in  $E_2$ .

**OBR for Quantization.** Different from pruning where the retain set and eviction set can be naturally obtained from the pruning mask, in quantization, we need to manually assign the grouping to obtain  $R_2$  and  $E_2$  for compensation with OBR. Thanks to the flat distribution introduced by Hadamard rotation, we find the discrepancy among unpruned elements is actually small (see Fig. 6). Inspired by this observation, we propose to take the first  $\alpha$  proportion of elements in set  $R_1$  as the set  $E_2$ , and the remaining  $1 - \alpha$  proportion of elements as the set  $R_2$ . In other words,  $|R_2| + |E_2| = |R_1|$ , where  $|\cdot|$  is the number of elements. In Fig. 3(c), given quantization error on set  $E_2$  as  $\mathbf{e}_{E_2}^{quant} = \bar{\mathbf{w}}_{E_2} - \mathrm{quant}(\bar{\mathbf{w}}_{E_2})$ , we can obtain the OBR compensation for quantization as follows:

$$\Delta \mathbf{w}_{R_2}^{quant} = -\mathbf{H}_{R_2 R_2}^{-1} \mathbf{H}_{R_2 E_2} (\bar{\mathbf{w}}_{E_2} - \text{quant}(\bar{\mathbf{w}}_{E_2})). \tag{10}$$

Considering both quantization and pruning, the overall OBR-processed weights can be formalized as:

$$\hat{\mathbf{w}} = \operatorname{quant}([\mathbf{w}_{R_2} + \Delta \mathbf{w}_{R_2}^{prune} + \Delta \mathbf{w}_{R_2}^{quant}, \quad \mathbf{w}_{E_2} + \Delta \mathbf{w}_{E_2}^{prune}, \quad \mathbf{0}]), \tag{11}$$

where  $\Delta \mathbf{w}_{R_2}^{prune}$  and  $\Delta \mathbf{w}_{E_2}^{prune}$  denote indexing from  $\Delta \mathbf{w}_{R_1}^{prune}$  using  $R_2$  and  $\hat{\mathbf{w}}$  is the final joint low-bit and sparse weights. Algo. 1 provides more details of our proposed OBR.

**CUDA Kernel Implementation.** After transforming LLMs to both sparse and low-bit using the proposed OBR, we implement corresponding GEMM with the CUTLASS library<sup>1</sup>. Due to hardware support limitations, we perform 2:4 semi-structured sparsity and INT4 quantization on the weights **W**, and use INT4 quantization for the activations **X**. Related experiments are shown in Sec. 5.1.

