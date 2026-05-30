# 2 Revisiting MoE-based LoRA Methods

#### 2.1 Preliminaries

LoRA (visualized in Figure 2(a)) simulates weight updates during fine-tuning by decomposing the update matrix into two learnable low-rank matrices. Given a pretrained weight matrix  $W_0 \in \mathbb{R}^{m \times n}$ , the parameter update is computed as:

$$\mathbf{W}' = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A},\tag{1}$$

where  $\boldsymbol{B} \in \mathbb{R}^{m \times r}$ ,  $\boldsymbol{A} \in \mathbb{R}^{r \times n}$ , and the rank  $r \ll \min(m, n)$ . The scaling factor  $\alpha$  is typically set to 2r. For an input embedding  $\boldsymbol{x} \in \mathbb{R}^n$ , the forward pass becomes:

<span id="page-2-0"></span>
$$f_{\text{LoRA}}(\boldsymbol{x}) = \boldsymbol{W}' \boldsymbol{x} = \boldsymbol{W}_0 \boldsymbol{x} + \frac{\alpha}{r} \boldsymbol{B} \boldsymbol{A} \boldsymbol{x}.$$
 (2)

Here,  $W_0$  remains frozen during training, while only  $\{A, B\}$  are updated. This approach reduces the number of trainable parameters from  $\mathcal{O}(mn)$  to  $\mathcal{O}(r(m+n))$ , thereby achieving higher parameter efficiency. The low-rank structure allows LoRA to maintain stable performance while significantly reducing both computational overhead and GPU memory requirements during fine-tuning.

#### <span id="page-2-2"></span>2.2 MoE-based LoRA Framework

The MoE paradigm (visualized in Figure 2(b)) extends LoRA by decomposing the low-rank adaptation into N specialized experts. Each expert  $E_i$  is parameterized by a pair of matrices  $\{B_i \in \mathbb{R}^{m \times r_i}, A_i \in \mathbb{R}^{r_i \times n}\}$ , where  $r_i$  denotes the expert-specific rank. The forward pass incorporates a gating mechanism  $G(x) : \mathbb{R}^n \to \mathbb{R}^N$  that dynamically routes inputs to activate the most relevant experts. Formally, the output combines the frozen pretrained weights  $W_0$  with a sparse combination of expert contributions:

<span id="page-2-1"></span>
$$f_{\text{MoE-LoRA}}(\boldsymbol{x}) = \boldsymbol{W}_0 \boldsymbol{x} + \frac{\alpha}{r} \sum_{i=1}^{N} \boldsymbol{G}(\boldsymbol{x})_i \cdot \underbrace{\boldsymbol{B}_i \boldsymbol{A}_i \boldsymbol{x}}_{\boldsymbol{E}_i(\boldsymbol{x})}, \tag{3}$$

where the router G(x) typically follows a top-k selection policy via a trainable projection  $W_g \in \mathbb{R}^{N \times n}$ . For simplicity, we omit the activation function, formulating the router as:

$$G(x) = top-k(W_q x). (4)$$

By activating only k experts per input, this design maintains computational efficiency. The sparse routing strategy enables conditional computation, which expands the model's representational capacity without incurring a proportional increase in computational cost. In our work, we implement SplitLoRA under this framework as a minimal yet representative instantiation of MoE-based LoRA. Further implementation details are provided in Appendix  $\mathbb{C}.3$ .

<span id="page-3-0"></span>![](_page_3_Picture_0.jpeg)

Figure 2: Schematic illustrations of different LoRA variants. (a) LoRA employs low-rank matrices A and B to simulate weight updates, where each row of A is fully connected to the corresponding column of B. (b) MoE-based LoRA decomposes the updates into multiple small experts  $\{A_i, B_i\}_{i=1}^N$  and uses a router to determine which experts should be activated. (c) FlyLoRA unifies the down-projection and router into a frozen matrix A and selectively activates only the ranks in B linked to the top-k magnitude activations after projection through A.

#### <span id="page-3-2"></span>2.3 Pushing MoE-based LoRA Architecture to the Extreme

Comparing Eq. 2 and Eq. 3 reveals that MoE-based LoRA can be viewed as a finer-grained, sparsely activated variant of LoRA, where the separation of experts mitigates task conflicts. Taking this decomposition to the extreme motivates our **rank-wise expert** design, where each expert governs a single rank, achieving the best decorrelating effect (see Figure 1(a)). Formally, for a rank-r LoRA, the matrices A and B can be decomposed into r rank-1 components:

$$f_{\text{rank-wise-LoRA}}(\boldsymbol{x}) = \boldsymbol{W}_0 \boldsymbol{x} + \frac{\alpha}{r} \sum_{i=1}^{r} \boldsymbol{G}(\boldsymbol{x})_i \cdot \underbrace{\boldsymbol{b}_i \boldsymbol{a}_i \boldsymbol{x}}_{\boldsymbol{E}_i(\boldsymbol{x})},$$
 (5)

with 
$$a_i = A[i,:] \in \mathbb{R}^{1 \times n}$$
 and  $b_i = B[:,i] \in \mathbb{R}^{m \times 1}$ .

However, this approach introduces a scalability challenge: the router's linear layer  $W_g \in \mathbb{R}^{N \times n}$  grows linearly with the number of experts N (see Figure 1(b)). Under a fixed total rank, finer-grained experts with larger N make the explicit routing mechanism computationally prohibitive, undermining the efficiency gains of sparse activation.

To overcome this limitation, we seek an **implicit routing mechanism** that eliminates the need for the explicit router parameter  $W_g$  entirely. This entails finding a proxy that leverages intrinsic signals within the model to select the top-k experts, effectively approximating the function of the original router G. To address this, we draw inspiration from the perspective of Singular Value Decomposition (SVD), which can also be viewed as a rank-wise decomposition. In SVD, the low-rank update matrix  $\Delta W$  can be decomposed as  $\Delta W = \sum_{i=1}^r \sigma_i u_i v_i^{\mathsf{T}}$ , where  $\sigma_i$  denotes the i-th singular value (indicating the importance of the corresponding component),  $u_i$  is the i-th left-singular vector, and  $v_i$  is the i-th right-singular vector. Each component  $\sigma_i u_i v_i^{\mathsf{T}}$  is a rank-1 update. The Eckart-Young-Mirsky theorem [18] guarantees that the top-k components, selected based on the magnitude of  $\sigma_i$ , provide the best rank-k approximation to the original rank-r matrix in terms of Frobenius norm, thereby capturing the most salient features with minimal reconstruction error. While exact SVD is computationally prohibitive and thus impractical in our framework, this insight naturally suggests that the magnitude of each rank-1 term,  $\|b_i a_i x\|$  in Eq. 6, approximately reflects its importance:

<span id="page-3-1"></span>
$$f_{\text{LoRA}}(\boldsymbol{x}) = \boldsymbol{W}_0 \boldsymbol{x} + \frac{\alpha}{r} \sum_{i=1}^{r} \boldsymbol{b}_i \boldsymbol{a}_i \boldsymbol{x}.$$
 (6)

Nevertheless, a naive approach of first computing all r terms  $b_i a_i x$  and then selecting the top-k would also forfeit the computational benefits of sparse activation, as the cost of computing all terms remains  $\mathcal{O}(rmn)$ . This necessitates a routing strategy that can identify the most important experts before fully computing their outputs. Furthermore, beyond efficient routing, another critical limitation of existing MoE-based LoRA methods is their lack of inherent support for multi-task deployment.

When merging models already fine-tuned on different tasks, interference between LoRA adapters often leads to significant performance degradation, as the underlying architecture does not structurally encourage task-specific updates to reside in orthogonal or non-overlapping parameter subspaces.

These dual challenges motivate the following two key design requirements for an improved MoE-based LoRA framework:

- Implicit magnitude-based router for top-k activation, without explicit router parameters, enabling expert selection prior to full computation;
- Native support for training-free model merging through architectural properties that promote inter-task interference mitigation.

