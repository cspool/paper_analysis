# <span id="page-4-2"></span>3 FlyLoRA

Inspired by the fly olfactory circuit (Figure 1(c)), whose neural architecture inherently meets our requirements for MoE-based LoRA variants, we propose FlyLoRA (visualized in Figure 2(c)). Section 3.1 presents its formal design, while subsequent sections analyze its key advantages: Section 3.2 shows how a fixed A acts as an implicit router, Section 3.3 demonstrates intra-task decoupling, and Section 3.4 establishes inherent support for inter-task decoupling in model merging.

#### <span id="page-4-0"></span>3.1 Formulation of FlyLoRA

In FlyLoRA, the matrix  $A \in \mathbb{R}^{r \times n}$  is sparse and frozen. It is randomly initialized at the beginning and remains frozen during training, implementing an intrinsic top-k operation in the projection space  $\mathbb{R}^r$  for implicit routing. Given an input token  $\mathbf{x} \in \mathbb{R}^n$ , this process is formulated as:

$$\mathbf{y}' = \operatorname{top-}k(\mathbf{y}) = \operatorname{top-}k(\mathbf{A}\mathbf{x}), \tag{7}$$

where each row of  $\boldsymbol{A}$  contains exactly p (p < n) non-zero entries independently sampled from  $\mathcal{N}(0,\frac{1}{r^2})$  (a widely used standard initialization). We define the sparsity ratio as  $\rho = \frac{p}{n}$ . After projection through  $\boldsymbol{A}$ , only the columns  $\boldsymbol{b}_i \in \mathbb{R}^m$  ( $i \in \{1,\ldots,r\}$ ) in the up-projection matrix  $\boldsymbol{B} \in \mathbb{R}^{m \times r}$  linked to dimensions with top-k (k < r) magnitudes in  $\boldsymbol{A}\boldsymbol{x} \in \mathbb{R}^r$  are activated. Formally:

$$[\mathbf{B}\mathbf{y}']_i = \begin{cases} [\mathbf{B}\mathbf{y}]_i & \text{if the magnitude of } [\mathbf{y}]_i \text{ is among the top-}k \text{ values of } \mathbf{y}, \\ 0 & \text{otherwise.} \end{cases}$$
(8)

To enhance training stability in this MoE structure, we incorporate a simple expert-wise bias term  $d \in \mathbb{R}^r$  for loss-free load balancing, following [43]. This auxiliary term is updated manually via:

$$d_i \leftarrow d_i + u \cdot \operatorname{sign}(\bar{c}_i - c_i),$$
 (9)

where u is a small learning rate,  $\bar{c}_i$  represents the expected assignment frequency for expert i,  $c_i$  tracks the actual assignment count, and  $\mathrm{sign}(\cdot)$  denotes the sign function. This bias term d is added to Ax in expert selection to promote the activation of under-activated experts and suppress over-activated experts, thereby achieving load balancing. Thus, the activated experts are selected by:

$$\mathcal{I}_{topk} = \{i_1, \dots, i_k\} \quad \text{where} \quad i_j = \underset{i \notin \{i_1, \dots, i_{j-1}\}}{\arg \max} \left(\mathbf{A}\mathbf{x} + \mathbf{d}\right)_i. \tag{10}$$

The forward pass is then computed as:

$$f_{\text{FlyLoRA}}(\boldsymbol{x}) = \boldsymbol{W}_0 \boldsymbol{x} + \Delta \boldsymbol{W} \boldsymbol{x} = \boldsymbol{W}_0 \boldsymbol{x} + \frac{\alpha}{r} \sum_{i=1}^r \mathbb{I}(i \in \mathcal{I}_{\text{top}k}) \cdot \boldsymbol{b}_i \boldsymbol{a}_i \boldsymbol{x}, \tag{11}$$

where  $\mathbb{I}(\cdot)$  denotes the indicator function.

