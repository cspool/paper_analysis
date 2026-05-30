# <span id="page-0-1"></span>FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

Seung-Wook Kim\* <sup>1</sup> , Seongyeol Kim\* <sup>1</sup> , Jiah Kim\* <sup>1</sup> , Seowon Ji† <sup>2</sup> , and Se-Ho Lee† <sup>3</sup> <sup>1</sup>Pukyong National University <sup>2</sup>Konkuk University <sup>3</sup> Jeonbuk National University

# Abstract

*Federated learning (FL) often suffers from performance degradation due to key challenges such as data heterogeneity and communication constraints. To address these limitations, we present a novel FL framework called FedWSQ, which integrates weight standardization (WS) and the proposed distribution-aware non-uniform quantization (DANUQ). WS enhances FL performance by filtering out biased components in local updates during training, thereby improving the robustness of the model against data heterogeneity and unstable client participation. In addition, DANUQ minimizes quantization errors by leveraging the statistical properties of local model updates. As a result, FedWSQ significantly reduces communication overhead while maintaining superior model accuracy. Extensive experiments on FL benchmark datasets demonstrate that FedWSQ consistently outperforms existing FL methods across various challenging FL settings, including extreme data heterogeneity and ultra-low-bit communication scenarios. The source code is available at our project page*[1](#page-0-0) *.*

# 1. Introduction

In large-scale machine learning, centralized approaches often raise privacy concerns because sensitive data from edge devices must be collected on a central server for model training. Federated learning (FL) [\[31\]](#page-9-0) enables distributed devices to collaboratively train a shared global model without sharing raw data. A fundamental FL method, FedAvg, was proposed by McMahan *et al*. [\[31\]](#page-9-0). In this approach, each client trains its model using the local data, and then the server collects and averages these local updates to form a unified global model.

Several studies on FL have verified its practical effectiveness [\[4,](#page-8-0) [28,](#page-8-1) [41,](#page-9-1) [42,](#page-9-2) [44,](#page-9-3) [49,](#page-9-4) [50\]](#page-9-5). However, real-world FL methods often face three key challenges [\[18,](#page-8-2) [19,](#page-8-3) [24,](#page-8-4) [30,](#page-9-6) [46\]](#page-9-7): 1) *data heterogeneity*, where clients possess nonindependent and identically distributed (non-*i.i.d.*) data; 2) *partial client participation*, where only a subset of clients contributes to a global model update during each training round due to communication constraints; 3) *communication bottlenecks*, where limited communication bandwidth and high transmission costs hinder efficient model aggregation. These challenges worsen local gradient divergence, slow down the global model convergence, and ultimately degrade FL performance in real-world applications.

To address these limitations, we propose FedWSQ, an advanced FL framework that combines weight standardization (WS) with our proposed distribution-aware nonuniform quantization (DANUQ) to improve both learning stability and communication efficiency. WS is a plug-andplay technique that standardizes the weight vectors of convolutional or linear layers to stabilize the learning process of a neural network [\[35\]](#page-9-8). In FedWSQ, WS plays a crucial role in mitigating client drift by filtering out gradient components that contribute to overfitting during local training. This leads to improved generalization across heterogeneous clients and significantly enhances FL performance. Although WS accelerates global convergence, the communication cost per round remains a major bottleneck. To address this, FedWSQ integrates WS with DANUQ, a novel quantization strategy that leverages a standard normal distribution prior to minimize quantization errors, reducing communication overhead while preserving model performance. Even under ultra-low-bit quantization, the proposed FedWSQ achieves superior performance, consistently outperforming state-of-the-art (SOTA) FL methods across various datasets and FL settings.

Our main contributions are summarized as follows.

- We propose an effective application of WS in FL, enhancing model convergence and stability under data heterogeneity and limited client participation. By implicitly performing a gradient filtering process, WS mitigates client drift while also providing a regularization effect that stabilizes training and improves generalization.
- We introduce DANUQ, a novel quantization method that employs a fixed quantization function, leveraging the statistical properties of local model parameter updates (LM-

<sup>\*</sup>indicates equal contribution.

<sup>†</sup>Corresponding authors.

<span id="page-0-0"></span><sup>1</sup><https://github.com/Seongyeol-kim/FedWSQ>

<span id="page-1-0"></span>PUs) to minimize quantization errors under parameterdistribution priors. By efficiently compressing LM-PUs while preserving essential information, the proposed DANUQ significantly reduces communication overhead without noticeable performance degradation.

 Extensive experiments show that FedWSQ significantly enhances performance across diverse settings. Even with an average of 2.3 bits per parameter, FedWSQ improves performance by over 5% on Tiny-ImageNet under a highly heterogeneous setting compared to SOTA FL methods.

#### 2. Related Work and Preliminaries

Federated learning (FL) FL is a decentralized approach that enables collaborative training of a global model. Using distributed devices, FL ensures efficient learning while preserving data privacy. Let  $\mathcal{F}_n(\mathbf{w}_g) := \mathbb{E}_{\zeta_n \sim \mathcal{D}_n} \left[ \mathcal{F}_n(\mathbf{w}_g; \zeta_n) \right]$  be the local loss function of the client  $n \in [N]$  with a local data distribution  $\mathcal{D}_n$ , where  $\mathbf{w}_g$  is the global model parameter (GMP) vector. The training objective function of FL [31] is formulated as follows:

$$\min_{\mathbf{w}_{g}} \Big\{ \mathcal{F}(\mathbf{w}_{g}) := \sum_{i \in \mathcal{S}} h_{i} \mathcal{F}_{i}(\mathbf{w}_{g}) \Big\}, \tag{1}$$

where  $h_i$  is the weight assigned to the i-th device, such that  $h_i \geq 0$ ,  $\sum_i h_i = 1$ , and  $h_i \propto |\mathcal{D}_i|$ , and  $\mathcal{S} \subseteq [N]$  is a subset of participating clients in the current communication round. At each round  $t \in [T]$ , the server broadcasts the GMPs  $\mathbf{w}_{\mathbf{g}}^{t-1}$  to the participating client i. In FedAvg [31], the i-th client performs K steps of local training to obtain the local model parameter (LMP) vector,  $\mathbf{w}_i^t$ , and transmits its LMP updates (LMPUs),  $\Delta \mathbf{w}_i^t \coloneqq \mathbf{w}_i^t - \mathbf{w}_{\mathbf{g}}^{t-1}$ , to the server. Finally, the server aggregates these LMPUs through weighted averaging to derive the updated GMPs.

In real-world FL scenarios, unstable learning dynamics and slow convergence often occur due to non-i.i.d. data distributions and limited communication bandwidth. To address these issues, FedProx [27] penalizes LMPUs by adding a proximal term in the local objective to reduce the gap between the global and local loss functions. FedAvgM [13] integrates momentum into the global update process to mitigate variance in model aggregation. FedAdam [36] introduces an adaptive optimizer that improves global and local model convergence, thereby accelerating overall training progress. FedDyn [1] presents dynamic regularization for each device to diminish deviations in the LMPUs caused by data heterogeneity. FedSmoo [43] applies constraints that prevent sharp fluctuations in local updates to generalize FL performance by ensuring global consistency. Recently, FedACG [21] was proposed to improve convergence by broadcasting a global model with a lookahead gradient and aligning local models with the shared global model.

**Network quantization** As neural network architectures have grown larger in scale, quantization has become essential for the memory optimization and acceleration of operations [9, 10, 16, 17, 29, 33, 55]. Let us consider a model using B-bit representation. A typical quantization function  $\mathcal Q$  mapping a full-precision value x to a quantization level (QL) q can be written as

$$q = \mathcal{Q}(x/s),\tag{2}$$

where s is a scaling factor, and each QL is matched with a B-bit integer from  $\{0,\ldots,2^B-1\}$ . Note that  $\mathcal{Q}(\cdot)$  assigns an input value to one of  $2^B$  QLs through nearest-neighbor clustering. The primary goal of effective quantization is to determine the proper scaling factor and a set of QLs to minimize the quantization errors.

Quantization methods can be roughly categorized into two approaches: uniform quantization (UQ) and nonuniform quantization (NUQ). UQ divides the input range into  $2^B$  equal intervals and assigns a representative value as a QL to each interval [10, 29]. In contrast, NUQ adjusts the interval widths based on the prior data distribution [6, 8, 51]. By allocating finer intervals in dense data regions and broader intervals in sparse regions, NUQ preserves critical information more effectively than UQ. For the scaling factor s, both approaches commonly employ the absmax method [16, 33] to determine the dynamic range for quantization. Furthermore, to enhance data adaptability, researchers have explored methods for dynamically learning quantization parameters (e.g., scaling factors and QLs) [9, 17, 55]. During backpropagation, these parameters are updated to reduce quantization error and better align with the underlying data distribution.

**Quantized FL** In FL, communication bottlenecks arise primarily during uplink transmission, as each participating client must send its individual LMPUs to the central server. Unlike downlink transmission, where a single GMP is broadcast to multiple clients, uplink communication involves substantial data transfer. This problem is exacerbated in resource-constrained communication environments, such as mobile and IoT networks, where upload bandwidth is severely limited [5, 37]. To overcome this challenge, several studies have focused on compressing model updates. In [22], quantization and sparsification are combined to reduce the number of transmitted parameters. FedPAQ [37] employs a probabilistic rounding algorithm to prevent an excessive concentration of identical QLs. Meanwhile, FedHQ+ [5] assigns different weight factors to clients for global updates based on their individual quantization errors. Although these methods significantly improve communication efficiency, they often suffer from performance degradation under extremely low bitwidth conditions. In this paper, we introduce FedWSQ, a

<span id="page-2-3"></span><span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1. The overall framework of FedWSQ, which integrates WS and DANUQ for efficient FL. Clients transmit LMPUs quantized by the DANUQ along with scaling vectors to the server, where dequantization and aggregation are performed to update the global parameters.

novel quantization framework that is designed to maintain robust model performance even at ultra-low bit-width.

### 3. Proposed Method

Figure 1 illustrates the overall framework of the proposed FedWSQ, which employs WS and DANUQ for efficient FL. Each client has a standard neural network model consisting of multiple model blocks, where WS is applied to mitigate the impact of data heterogeneity between clients. After local training, the full-precision values of LMPUs are converted to a B-bit representation using the global scaling vectors and carefully designed QLs. These quantized LM-PUs and their corresponding scaling vectors are transmitted to the server. The server then dequantizes the received data to higher-precision values and aggregates them to update the GMPs. In addition, the global scaling vectors are refined to ensure consistent scaling across different clients for DANUQ. Using the proposed DANUQ, FedWSQ enhances communication efficiency without sacrificing model performance in challenging FL scenarios.

#### 3.1. Weight standardization (WS)

WS [35] standardizes each weight vector of a convolution or linear layer to enhance the learning process. By ensuring consistent parameter distributions during training, WS contributes to a more stable gradient flow. Beyond this general benefit, we argue that the use of WS in FL can address the challenges posed by the distributed nature of FL and data heterogeneity. Formally, consider the n-th client that has a network architecture consisting of L linear or convolutional layers. Let the *l*-th layer (without bias) map an input  $\mathbf{x}_l \in \mathbb{R}^{I_l}$  to an output  $\mathbf{y}_l \in \mathbb{R}^{O_l}$ , defined by

$$\mathbf{y}_l = \mathbf{W}_{n,l}^T \mathbf{x}_l \tag{3}$$

$$l = \mathbf{W}_{n,l}^{T} \mathbf{x}_{l}$$

$$:= \begin{bmatrix} \mathbf{w}_{n,m_{1}^{l}} & \mathbf{w}_{n,m_{2}^{l}} & \cdots & \mathbf{w}_{n,m_{O_{l}}^{l}} \end{bmatrix}^{T} \mathbf{x}_{l},$$
(4)

where  $\mathbf{w}_{n,m}$ , for  $m \in \mathcal{M}_l := \{m_1^l, \dots, m_{O_l}^l\}$ , is the weight vector.<sup>2</sup> In WS, each  $\mathbf{w}_{n,m}$  is standardized before being used within the layer. We denote the prestandardized parameter (PSP) vector as  $\mathbf{w}_{n,m}$ , and the weight-standardized parameter (WSP) vector as  $\tilde{\mathbf{w}}_{n,m}$ , respectively. Then, the WSP vector  $\tilde{\mathbf{w}}_{n,m}$  is obtained by

<span id="page-2-2"></span>
$$\tilde{\mathbf{w}}_{n,m} = \frac{\rho}{\sigma(\mathbf{w}_{n,m})} \left( \mathbf{I} - \mathbf{P_1} \right) \mathbf{w}_{n,m}, \tag{5}$$

where  $\mathbf{1} \in \mathbb{R}^{I_l}$  is a vector of all ones,  $\sigma(\mathbf{w}_{n,m})$  is the standard deviation of the elements in  $\mathbf{w}_{n,m}$ , and  $\mathbf{P}_{\mathbf{v}}$  represents the projection matrix onto the subspace spanned by the vector  $\mathbf{v}$ , which is denoted by span $\{\mathbf{v}\}$ . Here, we introduce a hyper-parameter  $\rho$  to control the normalization scale. Note that we reformulate the WSP computation from [35] into Eq. (5) to better understand the gradient filtering process of WS during local training.

Gradient filtering by WS WS plays an important role in reducing the learning diversity of local models, which is one of the most critical issues in FL. To understand its impact, we compare the gradient of a loss function  $\mathcal{L}$  with respect to the WSP vector  $\tilde{\mathbf{w}}_{n,m}$  to that of the PSP vector  $\mathbf{w}_{n,m}$ . When WS is applied, the gradient is processed by an additional series of projections to remove specific components

<span id="page-2-1"></span><sup>&</sup>lt;sup>2</sup>Strictly speaking, convolution involves spatial structures aggregating information from local receptive fields. For simplicity, we omit spatial considerations and treat the convolution as a linear transformation. This abstraction can be easily generalized to multidimensional tensor structures.

<span id="page-3-4"></span><span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 2. An example of gradient filtering process of WS.

that may interfere with effective FL. Specifically, the gradient with respect to the PSP vector wn,m is obtained by[3](#page-3-0)

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_{n,m}} = \frac{\rho}{\sigma(\mathbf{w}_{n,m})} \left( \mathbf{I} - \mathbf{P}_{\mathbf{1}} \right) \left( \mathbf{I} - \mathbf{P}_{\tilde{\mathbf{w}}_{n,m}} \right) \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{w}}_{n,m}}.$$
(6)

The upstream gradient <sup>∂</sup><sup>L</sup> ∂w˜ n,m is projected onto two subspaces, span{w˜ n,m} <sup>⊥</sup> and span{1} <sup>⊥</sup>, where span{v} ⊥ represents the orthogonal complement of span{v}. Since w˜ n,m ⊥ 1, the upstream gradient <sup>∂</sup><sup>L</sup> ∂w˜ n,m is ultimately projected onto span{w˜ n,m, 1} <sup>⊥</sup> after the two projections.

As illustrated in Figure [2,](#page-3-1) Eq. [\(6\)](#page-3-2) represents a gradient filtering process that applies successive projections to remove undesirable components from the gradient with respect to LMPs. In FL, data heterogeneity causes local models to overfit their local data, which leads to inconsistent updates across clients. This inconsistency results in a discrepancy between the GMPs and the optimal parameters, known as client drift. WS mitigates this issue by projecting the gradient onto span{w˜ n,m, 1} <sup>⊥</sup>, which effectively reduces the impact of client drift during training. The first projection eliminates the gradient component aligned with w˜ n,m, which corresponds to the LMPs that are biased due to the non-*i.i.d.* data distribution. The second projection removes the mean component of mini-batch gradients, which can also be biased toward the local data distribution, compounding inconsistencies across clients. By filtering out both the parameter-aligned and mean components from the local gradient, WS preserves meaningful gradient directions essential for stable global model convergence. Additionally, this projection-based filtering introduces a regularization effect by constraining the model's deviation from its current state, which can improve the model's generalization ability.

<span id="page-3-3"></span>Comparison with FedWon It is worth noting that Fed-Won [\[56\]](#page-9-15), a recently proposed FL method, also employs WS to address data heterogeneity in multi-domain FL. However, a key difference lies in the parameters exchanged between clients and the server, which fundamentally alters the learning dynamics. FedWon transmits WSP, which explicitly normalizes parameters across clients to improve learning stability. However, this approach forces all clients' LMPUs to exhibit statistically similar characteristics, which potentially discards client-specific information crucial for local adaptation. In contrast, FedWSQ transmits PSP, preserving essential local information while implicitly mitigating harmful divergences through the gradient filtering process of WS, as shown in Figure [2.](#page-3-1) This design choice enables FedWSQ to achieve a better balance between global stability and local information preservation. An experimental comparison between FedWon and FedWSQ will be discussed in Section [4.4.](#page-6-0)

# 3.2. Quantization of LMPUs

<span id="page-3-2"></span>Neural network quantization often involves learning quantization parameters. However, in FL, transmitting additional quantization parameters or auxiliary information for each communication round would introduce significant resource overhead. Instead, we propose a fixed quantization function that eliminates the need to transmit additional quantization information, ensuring a lightweight and efficient communication process. A fundamental assumption in NUQ is that model parameters follow a normal distribution, as demonstrated in prior studies [\[7,](#page-8-18) [45,](#page-9-16) [54\]](#page-9-17). Since LMPUs represent the difference between an updated LMP and the previous GMP, they may also exhibit a normal distribution.

LMPU scaling Most quantization methods adopt the *absmax* strategy to manage the dynamic range of values [\[16,](#page-8-11) [33\]](#page-9-11). This approach scales an input tensor to the range [−1, 1] by dividing each element by the maximum absolute value of the tensor. Within this constrained range, conventional quantization methods determine QLs and map input values to the nearest quantization point. However, these methods are highly sensitive to outliers, which can excessively expand the dynamic range of the target tensor. Such over-scaling often results in underflow, especially in extremely low-bit representation. To solve this problem, we use the standard deviation of the input tensor as a scaling factor. Unlike the *absmax* value, the standard deviation is more robust to outliers. Moreover, since our DANUQ is based on a probability model of a standard normal distribution, the standard deviation provides a more reliable approximation of the target dynamic range.

In FedWSQ, we use a shared global scaling factor to help maintain consistency in quantization. Instead of independently using scaling factors, we employ a collaborative approach in which each client contributes to a global scaling vector, ensuring that local standard deviations are taken into account. Specifically, each client i computes a scaling vector, defined as s<sup>i</sup> = [si,1, . . . , si,L] T , where si,l denotes the standard deviation of the LMPUs for the l-th layer. The server then collects and aggregates s<sup>i</sup> from the participating clients i ∈ S, and updates the global scaling vector

<span id="page-3-0"></span><sup>3</sup>Section [A](#page-10-0) of the supplementary document provides its derivation.

$$\mathbf{s}_{\mathsf{g}} = [s_{\mathsf{g},1}, \dots, s_{\mathsf{g},L}]^T$$
 as follows:

$$\mathbf{s}_{g} \leftarrow (1 - \beta)\mathbf{s}_{g} + \beta \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \mathbf{s}_{i},$$
 (7)

where  $\beta$  is a momentum parameter that controls the update rate. This approach ensures that the global scaling factor smoothly adapts while mitigating local fluctuations. Once updated, the server distributes  $\mathbf{s}_{g}$  to the participating clients. Each client then divides its LMPU updates for the l-th layer by the corresponding scale  $s_{g,l}$ , so that the normalized values can be regarded as samples from a standard normal distribution. Based on this assumption, we apply the proposed DANUQ scheme detailed in the next section.

**QLs considering normal distribution** To determine the optimal multi-bit QLs, we formulate an optimization problem that minimizes the expected quantization error under a normal distribution. We assume that a random variable  $\Delta w$  from the normalized LMPUs follows a standard normal distribution. Since the distribution is symmetric around zero, we consider only its nonnegative half. Given a B-bit representation, let  $Q = \{q_0, q_1, \dots, q_R\}$  be the set of QLs, where  $R = 2^{B-1} - 1$ . Here, we assume that  $q_r$  values are sorted in ascending order and set  $q_0 = 0$  as a fixed value. Then, the quantization boundaries are defined as

$$u_r = \begin{cases} q_0, & \text{if } r = 0, \\ \frac{q_{r-1} + q_r}{2}, & \text{if } 1 \le r \le R, \\ +\infty, & \text{if } r = R + 1. \end{cases}$$
 (8)

A full-precision value in the range  $[u_r, u_{r+1})$  is quantized into the QL  $q_r$ . The optimal QLs are determined by minimizing the expected quantization error, formulated as

<span id="page-4-0"></span>
$$\mathbb{E}\left[(\Delta w - \Delta \bar{w})^2\right] = \sum_{r=0}^{R} \int_{u_r}^{u_{r+1}} (x - q_r)^2 p(x) dx, \quad (9)$$

where  $\Delta \bar{w} \in Q$  represents the quantized value, and p(x) is the PDF of the standard normal distribution. From (9), we can derive the following optimization problem as

$$\min_{\{q_1,\dots,q_R\}} \left\{ \frac{1}{2} (q_R^2 + 1) - \sqrt{\frac{2}{\pi}} q_1 e^{-\frac{u_1^2}{2}} - \frac{1}{2} q_1^2 \text{erf} \left(\frac{u_1}{\sqrt{2}}\right) + \sqrt{\frac{2}{\pi}} \sum_{r=1}^{R-1} (q_r - q_{r+1}) e^{-\frac{u_{r+1}^2}{2}} + \frac{1}{2} \sum_{r=1}^{R-1} (q_r^2 - q_{r+1}^2) \text{erf} \left(\frac{u_{r+1}}{\sqrt{2}}\right) \right\}, \tag{10}$$

where  $\operatorname{erf}(\cdot)$  denotes the error function.<sup>4</sup> A closed-form solution to Eq. (22) is intractable due to the presence of nonlinear terms, including the Gaussian integral and the error

#### Algorithm 1: FedWSQ

return  $(\Delta \overline{\mathbf{W}}_i, \mathbf{s}_i)$ 

```
Input: Initial global parameters \mathbf{W}_{g}^{0}, initial scales \mathbf{s}_{g}^{0}, global
                 communication round T, local training iteration K, number
                 of total clients N, number of model layers L, learning rate
                \eta, hyperparameter \beta
Server-side:
for t=1,\ldots,T do
           Randomly sample a subset of clients S_t \subseteq [N]
           Set the bit-precision B_i for the client i \in \mathcal{S}_t
           Send (\mathbf{W}_{\mathrm{g}}^{t-1}, \mathbf{s}_{\mathrm{g}}^{t-1}) to client i \in \mathcal{S}_t
           for each client i \in \mathcal{S}_t, in parallel do
                     (\Delta \overline{\mathbf{W}}_i^t, \mathbf{s}_i^t) \leftarrow
                         ClientLocalTraining(\mathbf{W}_{\mathrm{g}}^{t-1}, \mathbf{s}_{\mathrm{g}}^{t-1}, i, B_{i})
           end
           for i \in \mathcal{S}_t do
                  Obtain dequantized parameters \Delta_i^t using (\Delta \overline{\mathbf{W}}_i^t, \mathbf{s}_i^t)
          Aggregate local updates: \Delta^t \leftarrow \sum_{i \in \mathcal{S}_t} \Delta^t_i
Update global parameters: \mathbf{W}_{\mathbf{g}}^t \leftarrow \mathbf{W}_{\mathbf{g}}^{t-1} + \Delta^t
Update global scales: \mathbf{s}_{\mathbf{g}}^t \leftarrow (1-\beta)\mathbf{s}_{\mathbf{g}}^{t-1} + \beta\frac{1}{|\mathcal{S}_t|}\sum_{i \in \mathcal{S}_t}\mathbf{s}_i^t
 \begin{aligned} \textbf{ClientLocalTraining}(\mathbf{W}_{\mathrm{g}}, \mathbf{s}_{\mathrm{g}}, i, B) \colon \\ \text{Initialize local parameters: } \mathbf{W}_{i}^{0} \leftarrow \mathbf{W}_{\mathrm{g}} \end{aligned} 
Set DANUQ function \mathcal{Q}(\cdot) based on the bit-precision B
for k = 1, ..., K do

Apply WS to \mathbf{W}_i^{k-1} using (5)

Compute unbiased gradient of local loss \nabla f_i(\mathbf{W}_i^{k-1})

Update local parameters: \mathbf{W}_i^k \leftarrow \mathbf{W}_i^{k-1} - \eta \nabla f_i(\mathbf{W}_i^{k-1})
Compute local update: \Delta \mathbf{W}_i \leftarrow \mathbf{W}_i^K - \mathbf{W}_{\mathrm{g}}
for each layer l=1,\ldots,L do
           Compute quantized local update: \Delta \overline{\mathbf{W}}_{i,l} \leftarrow \mathcal{Q}(\Delta \mathbf{W}_{i,l}/s_{g,l})
           Calculate local scale factors: s_{i,l} \leftarrow \operatorname{std}(\Delta \mathbf{W}_{i,l})
end
 \Delta \overline{\mathbf{W}}_i \coloneqq (\Delta \mathbf{W}_{i,1}, \cdots, \Delta \mathbf{W}_{i,L})
 \mathbf{s}_i \coloneqq [s_{i,1}, \cdots, s_{i,L}]
```

function, which lack simple analytical inverses, and the interdependent QLs due to boundary conditions. As an alternative, we employ a brute-force search algorithm to numerically determine the optimal QLs.<sup>5</sup> The optimal QLs for different bit-widths are obtained as follows: 1) 1-bit<sup>6</sup>: [-0.798, 0.798]; 2) 2-bit: [-1.224, 0, 0.765, 1.724]; 3) 4-bit: [-2.654, -1.974, -1.508, -1.149, -0.834, -0.544, -0.269, 0, 0.230, 0.465, 0.708, 0.966, 1.248, 1.568, 1.968, 2.649]. These QLs minimize quantization error under the normal distribution, which provides an effective trade-off between precision and computational efficiency.

<span id="page-4-1"></span><sup>&</sup>lt;sup>4</sup>The full derivation is provided in the supplementary document.

<span id="page-4-2"></span><sup>&</sup>lt;sup>5</sup>We discretize the search space of possible QLs and perform an exhaustive evaluation of the cost function to identify the configuration that minimizes the quantization error. To improve computational efficiency, we restrict the search space to a reasonable range based on empirical observations and employ parallel processing to accelerate evaluation.

<span id="page-4-3"></span><sup>&</sup>lt;sup>6</sup>For the 1-bit case, the QLs consist of only two values. Thus, the constraint,  $q_0 = 0$ , is omitted to allow optimal placement of both QLs.

<span id="page-5-1"></span><span id="page-5-0"></span>

| Table 1. FL performance on the three benchmark datasets with a 5% participation rate over 100 clients for $\alpha \in \{0.1, 0.3\}$ |
|-------------------------------------------------------------------------------------------------------------------------------------|
|-------------------------------------------------------------------------------------------------------------------------------------|

| Method         | #bits      | CIFAR-10       |       |       |        | CIFAR-100 |       |       | Tiny-ImageNet |       |       |       |        |
|----------------|------------|----------------|-------|-------|--------|-----------|-------|-------|---------------|-------|-------|-------|--------|
|                | 7010       | $\alpha = 0.1$ | 0.3   | 0.6   | i.i.d. | 0.1       | 0.3   | 0.6   | i.i.d.        | 0.1   | 0.3   | 0.6   | i.i.d. |
| FedAvg [31]    | 32         | 70.65          | 82.52 | 85.45 | 89.25  | 43.45     | 47.39 | 49.04 | 48.66         | 30.37 | 34.65 | 36.54 | 37.39  |
| FedProx [27]   | 32         | 70.89          | 83.91 | 86.60 | 88.81  | 43.78     | 48.26 | 48.49 | 48.54         | 29.99 | 35.44 | 37.14 | 37.43  |
| FedAvgM [13]   | 32         | 78.30          | 85.27 | 87.94 | 90.00  | 47.70     | 52.04 | 53.00 | 53.40         | 32.00 | 37.63 | 39.66 | 41.02  |
| FedADAM [36]   | 32         | 72.58          | 81.73 | 85.97 | 88.45  | 46.83     | 53.43 | 55.96 | 57.93         | 34.77 | 39.97 | 43.27 | 46.26  |
| FedDyn [1]     | 32         | 83.63          | 88.28 | 89.20 | 91.08  | 53.83     | 56.40 | 56.90 | 56.93         | 37.32 | 40.92 | 42.78 | 42.65  |
| FedMLB [21]    | 32         | 71.12          | 82.40 | 86.23 | 89.25  | 49.72     | 54.16 | 55.17 | 56.45         | 33.45 | 39.06 | 40.55 | 41.46  |
| FedLC [52]     | 32         | 74.03          | 83.80 | 86.20 | 88.35  | 43.86     | 48.50 | 48.61 | 47.81         | 31.73 | 34.55 | 36.72 | 38.06  |
| FedNTD [26]    | 32         | 73.29          | 83.34 | 86.26 | 89.36  | 45.08     | 48.73 | 50.39 | 50.90         | 34.36 | 36.44 | 39.06 | 40.60  |
| FedSmoo [43]   | 32         | 78.61          | 80.07 | 85.03 | 86.48  | 42.02     | 44.97 | 45.32 | 49.43         | 23.09 | 27.14 | 29.70 | 34.96  |
| FedDecorr [39] | 32         | 75.44          | 83.01 | 84.49 | 88.40  | 44.05     | 49.35 | 50.44 | 49.40         | 31.12 | 34.13 | 35.59 | 36.36  |
| FedWon [56]    | 32         | 70.89          | 78.36 | 82.44 | 86.56  | 45.55     | 51.68 | 53.60 | 55.28         | 30.52 | 36.20 | 38.59 | 40.73  |
| FedRCL [38]    | 32         | 83.45          | 88.19 | 89.74 | 91.55  | 58.26     | 62.21 | 63.95 | 65.11         | 37.86 | 44.94 | 46.75 | 47.96  |
| FedACG [20]    | 32         | 83.62          | 89.12 | 90.38 | 91.75  | 58.14     | 62.80 | 61.85 | 62.46         | 39.75 | 45.46 | 48.22 | 50.49  |
| E ID40 (27)    | 4          | 66.81          | 78.52 | 81.36 | 85.46  | 38.06     | 42.42 | 43.46 | 44.35         | 28.54 | 31.93 | 33.56 | 34.98  |
| FedPAQ [37]    | 1          | 57.82          | 71.27 | 72.36 | 79.46  | 30.07     | 34.16 | 38.24 | 37.85         | 24.74 | 29.48 | 30.74 | 33.81  |
| E-4110 . [5]   | 4          | 66.89          | 78.42 | 81.42 | 85.54  | 38.22     | 42.66 | 43.10 | 44.86         | 28.59 | 32.18 | 34.24 | 35.69  |
| FedHQ+ [5]     | 1          | 58.14          | 70.68 | 73.16 | 79.04  | 31.11     | 34.11 | 37.50 | 39.17         | 24.79 | 29.18 | 31.14 | 33.50  |
|                | FedWS (32) | 85.94          | 89.81 | 91.00 | 92.47  | 64.14     | 68.19 | 69.31 | 70.14         | 47.05 | 52.84 | 54.23 | 55.26  |
| FedWSQ         | 4          | 86.84          | 89.84 | 91.11 | 92.21  | 64.34     | 67.31 | 68.41 | 68.53         | 46.51 | 50.49 | 51.96 | 52.13  |
|                | 1          | 84.79          | 88.62 | 89.49 | 91.15  | 62.05     | 65.14 | 66.19 | 66.11         | 45.11 | 49.93 | 51.11 | 51.77  |
|                | FBA (2.33) | 86.16          | 89.60 | 90.44 | 91.71  | 63.07     | 66.16 | 67.08 | 67.17         | 46.62 | 50.69 | 51.43 | 51.68  |
|                | DBA (2.33) | 85.20          | 89.23 | 90.26 | 91.78  | 63.28     | 66.32 | 67.26 | 67.22         | 46.50 | 50.24 | 51.55 | 51.80  |

Mixed channel-bit allocation To further enhance communication efficiency in FL, we propose an adaptive mixedprecision strategy in which each client dynamically selects its bit representation based on its communication channel conditions. This approach effectively balances model update quality and transmission cost, reducing overall bandwidth utilization. To validate this strategy, we conduct two simulation setups: 1) Fixed-bit allocation (FBA), where each client keeps a constant bit-width selected from  $\{1,2,4\}$  throughout training, and 2) dynamic-bit allocation (DBA), where each client is assigned a randomly selected bit-width at every communication round. With bit-width selection following a uniform distribution over  $\{1, 2, 4\}$ , the expected bit-width per client is approximately 2.3 bits. The entire FedWSQ framework, including the mixed-precision strategy, is summarized in Algorithm 1.

#### 4. Experiments

In this section, we evaluate the proposed FedWSQ on standard FL benchmarks, compare its performance against various FL methods on both *i.i.d.* and non-*i.i.d.* data conditions, and offer a further analysis of FedWSQ. The convergence plots obtained from the experiments are provided in Section D of the supplementary document.

#### 4.1. Experimental setup

For experiments, we use three standard benchmark datasets: CIFAR-10 [23], CIFAR-100 [23], and Tiny-ImageNet [25]. CIFAR-10 and CIFAR-100 contain 60,000 images divided into 10 and 100 classes, respectively. Tiny-ImageNet is composed of 200 classes, providing a more complex image classification task. For *i.i.d.* settings, training sam-

ples are randomly assigned to clients without replacement. For the non-i.i.d. setting, we sample label ratios from a Dirichlet distribution with a concentration parameter  $\alpha \in$  $\{0.1, 0.3, 0.6\}$ , following Hsu et al. [13]. Lower values of  $\alpha$  (e.g., 0.1) indicate higher data heterogeneity, whereas larger values (e.g., 0.6) and the i.i.d. case correspond to more homogeneous distributions. The participation rate is set to 5% among 100 clients. Model accuracy is evaluated on the test set of each dataset at the 1,000th communication round. To ensure stable evaluations, we report accuracy using an exponential moving average with a smoothing parameter of 0.9. To validate the effectiveness of FedWSO, we integrate it with FedAvg [31]. Note that FedWSQ can be incorporated into any existing FL algorithm. For clarity, we refer to the proposed method without DANUQ as FedWS. For quantized FL settings, we consider three bitallocation strategies: constant allocation, FBA, and DBA. In the constant allocation settings, a fixed bit-width is assigned to all clients. We compare FedWSQ against various FL algorithms, including FedAvg [31], FedProx [27], FedAvgM [13], FedADAM [36], FedDyn [1], FedMLB [21], FedLC [52], FedNTD [26], FedSmoo [43], FedDecorr [39], FedWon [56], FedRCL [38], FedACG [20], FedPAQ [37], and FedHQ+ [5].

#### 4.2. Implementation details

We follow most of the implementation setups and evaluation protocols from [1, 20, 38, 48]. The ResNet-18 architecture [11] is adopted as the backbone network. Consistent with standard practice in FL [12], all BN [15] layers in ResNet-18 are replaced with GN [47] layers. Following the recommendation of [35], WS is applied before each GN

<span id="page-6-4"></span><span id="page-6-1"></span>

| Table 2. Comparison of UQ and DANUQ methods with or without WS on the three benchmark datasets with a 5% participation rate over |
|----------------------------------------------------------------------------------------------------------------------------------|
| 100 clients for $\alpha \in \{0.1, 0.3, 0.6\}$ .                                                                                 |

| Dataset       | WS | #bits | $\alpha = 0.1$ |       | $\alpha = 0.3$ |       | $\alpha = 0.6$ |       | i.i.d. |       |
|---------------|----|-------|----------------|-------|----------------|-------|----------------|-------|--------|-------|
|               |    |       | UQ             | DANUQ | UQ             | DANUQ | UQ             | DANUQ | UQ     | DANUQ |
|               |    | 1     | 57.82          | 72.48 | 71.27          | 80.88 | 72.36          | 83.52 | 79.46  | 86.92 |
|               | ×  | 2     | 62.39          | 65.19 | 74.35          | 78.61 | 76.79          | 82.59 | 82.25  | 86.11 |
| CIFAR-10      |    | 4     | 66.81          | 72.87 | 78.52          | 84.62 | 81.36          | 86.26 | 85.46  | 88.66 |
| CHI III       |    | 1     | 75.48          | 84.79 | 80.15          | 88.62 | 82.53          | 89.49 | 85.37  | 91.15 |
|               | ~  | 2     | 78.36          | 85.92 | 83.02          | 88.98 | 84.10          | 90.35 | 86.95  | 91.50 |
|               |    | 4     | 82.78          | 86.84 | 86.37          | 89.84 | 87.58          | 91.11 | 89.87  | 92.21 |
|               |    | 1     | 30.07          | 40.51 | 34.16          | 46.62 | 38.24          | 48.67 | 37.85  | 48.55 |
|               | ×  | 2     | 33.52          | 40.58 | 37.66          | 45.07 | 40.47          | 46.31 | 40.90  | 46.06 |
| CIFAR-100     |    | 4     | 38.06          | 43.59 | 42.42          | 48.33 | 43.46          | 49.29 | 44.35  | 48.76 |
|               |    | 1     | 45.59          | 62.05 | 50.10          | 65.14 | 52.13          | 66.19 | 54.58  | 66.11 |
|               | ~  | 2     | 49.68          | 63.34 | 54.22          | 65.96 | 56.01          | 67.19 | 58.72  | 67.82 |
|               |    | 4     | 56.85          | 64.34 | 60.21          | 67.31 | 61.98          | 68.41 | 62.98  | 68.53 |
|               |    | 1     | 24.74          | 28.95 | 29.48          | 33.14 | 30.74          | 36.75 | 33.81  | 38.95 |
| Tiny-ImageNet | ×  | 2     | 27.02          | 30.33 | 31.60          | 34.12 | 33.58          | 36.6  | 34.87  | 35.41 |
|               |    | 4     | 28.54          | 31.14 | 31.93          | 35.42 | 33.56          | 36.66 | 34.98  | 38.13 |
| ,gerver       |    | 1     | 32.73          | 45.11 | 37.63          | 49.93 | 39.60          | 51.11 | 41.98  | 51.77 |
|               | ~  | 2     | 36.16          | 45.96 | 40.90          | 50.17 | 42.51          | 51.12 | 44.89  | 51.12 |
|               |    | 4     | 40.46          | 46.51 | 44.58          | 50.49 | 45.55          | 51.96 | 46.93  | 52.13 |

<span id="page-6-2"></span>Table 3. Comparison of NUQ methods using WS on CIFAR-10 and CIFAR-100 datasets for  $\alpha \in \{0.1, 0.3, 0.6\}$ .

| #bits | Method | CI             | CIFAR-100   |             |             |             |             |
|-------|--------|----------------|-------------|-------------|-------------|-------------|-------------|
|       | curou  | $\alpha = 0.1$ | 0.3         | 0.6         | 0.1         | 0.3         | 0.6         |
| 1     | NF/FP  | 24.0           | 30.6        | 28.8        | 7.0         | 8.3         | 8.1         |
|       | DANUQ  | <b>84.8</b>    | <b>88.6</b> | <b>89.5</b> | <b>62.1</b> | 65.1        | 66.2        |
| 2     | NF     | 57.8           | 73.8        | 80.0        | 41.2        | 49.1        | 53.0        |
|       | FP     | 45.0           | 57.5        | 58.6        | 25.3        | 32.0        | 34.2        |
|       | DANUQ  | <b>85.9</b>    | <b>89.0</b> | <b>90.4</b> | <b>63.3</b> | <b>66.0</b> | <b>67.2</b> |
| 4     | NF     | 86.6           | 89.7        | 91.0        | 64.8        | 66.7        | 68.4        |
|       | FP     | <b>87.1</b>    | <b>90.0</b> | 91.0        | 64.3        | 67.0        | 68.4        |
|       | DANUQ  | 86.8           | 89.8        | <b>91.1</b> | 64.3        | <b>67.3</b> | 68.4        |

layer to achieve optimal performance. The hyper-parameter of WS is set to  $\rho=0.001$ , and the momentum for updating the scaling vector is set to  $\beta=0.1$ . Additional information on the hyper-parameter selection is provided in Section C of the supplementary document.

#### 4.3. Comparison with existing FL methods

Table 1 presents the performance comparison of the proposed FedWSQ against various existing FL methods on the three benchmark datasets. The bit-width settings for each FL method are indicated in the table. Compared to FL methods that use quantization, such as FedPAQ and FedHQ+, FedWSQ demonstrates superior performance for 4-bit and 1-bit settings. Also, FedWSQ outperforms FL methods with full-precision representation, including SOTA methods such as FedACG and FedRCL. Notably, even with a 1bit representation, FedWSQ demonstrates competitive performance on CIFAR-10 and achieves superior results on CIFAR-100 and Tiny-ImageNet. For example, in a highly heterogeneous setting ( $\alpha = 0.1$ ), FedWSQ (1-bit) achieves 62.05% accuracy on CIFAR-100, surpassing FedRCL and FedACG by 3.79% and 3.91%, respectively. Similarly, on Tiny-ImageNet, FedWSQ (1-bit) achieves 45.11% ac-

<span id="page-6-3"></span>![](_page_6_Picture_7.jpeg)

Figure 3. Loss landscape of FL methods trained on CIFAR-100, where the numbers indicate the Hessian top eigenvalue.

curacy, outperforming FedRCL and FedACG by 7.25% and 5.36%, respectively. These results highlight the effectiveness of FedWSQ, which demonstrates its robustness even in extremely low bit-width scenarios while maintaining high model accuracy.

#### <span id="page-6-0"></span>4.4. Additional analyses

Impact of bit representation in FedWSQ FedWSQ balances model performance and communication efficiency by adopting different bit-width settings. Table 1 presents the results for 32-bit (FedWS), 4-bit, and 1-bit representations, as well as adaptive bit-allocation strategies (FBA and DBA). Despite using lower bit-widths, FedWSQ remains robust in non-i.i.d. settings, particularly under high heterogeneity ( $\alpha = 0.1$ ). Since local models tend to overfit under high data heterogeneity, quantizing model updates helps to regularize overfitting, achieving more robust FL performance [2, 3]. The accuracy gaps among different bit-widths are relatively small, and thus the FBA and DBA strategies enable efficient communication with marginal performance loss. As shown in Table 1, both FBA and DBA strategies achieve performance close to the 4-bit setting while reducing communication costs by half. In summary, FedWSQ confirms its ability to maintain accuracy while significantly reducing communication bandwidth usage for real-world

<span id="page-7-3"></span><span id="page-7-1"></span>Table 4. FL performance of different backbone architectures for  $\alpha=0.3$ .

| Backbone        | Vanilla | FedWS        | FedWSQ (DBA) |
|-----------------|---------|--------------|--------------|
| ShuffleNet [53] | 36.37   | 53.32        | 51.12        |
| VGGNet-9 [40]   | 46.14   | 60.01        | 60.48        |
| SqueezeNet [14] | 39.45   | 55.87        | 56.10        |
| ResNet-18 [11]  | 47.39   | 68.19        | 66.32        |
| MobileViT [32]  | 35.42   | <u>40.80</u> | 41.27        |

FL scenarios.

Effect of WS and DANUQ Table 2 shows a comparative analysis of our proposed DANUQ and UQ under varying bit-widths and data heterogeneity. In this experiment, UQ is implemented using the FedPAQ approach [37]. As listed in the table, our DANUQ consistently outperforms the UQ method. Under extremely low-bit quantization (1bit and 2-bit settings), the UQ suffers from significant accuracy degradation. While WS alone improves performance, combining WS with the proposed DANUQ yields the highest performance gains. The UQ with WS shows improvements compared with its non-WS counterparts, but the synergy between WS and DANUQ provides the most robust FL performance. Furthermore, our method maintains high accuracy even with low-bit representations. This suggests that applying FBA or DBA can be highly effective, as our approach maintains strong performance across varying bitwidths and makes FedWSQ particularly practical for realworld FL deployments.

Comparison with other NUQ methods We evaluate the effectiveness of the proposed DANUQ by comparing it with existing NUQ approaches, FP [45] and NF [7]. As shown in Table 3, at the 4-bit representation, all NUQ methods perform competitively, indicating that NUQ is generally robust at relatively higher bit-widths. However, under 1-bit and 2-bit representations, FP and NF suffer from severe performance degradation, particularly in highly heterogeneous data ( $\alpha=0.1$ ). In contrast, our DANUQ outperforms both NF and FP across all conditions, which confirms its robustness against extreme quantization constraints. These results highlight the superiority of our DANUQ method among NUQ methods for communication-efficient FL.

Loss Landscape Analysis Figure 3 visualizes the loss landscapes of various FL methods compared to Fed-WSQ (with DBA). FedWSQ achieves the smoothest and most stable loss landscape, with the lowest Hessian top eigenvalue of 135.8, indicating improved generalization ability. FedSmoo exhibits a flatter landscape but converges to a much higher global minimum, reducing sharpness at the cost of performance. In contrast, FedACG reaches a global comparable to FedWSQ but has a significantly sharper landscape, with a Hessian top eigenvalue of 671.2, which indicates it is more sensitive to perturbations. These results

<span id="page-7-2"></span>Table 5. Ablation study for  $\rho$  using FedWS on CIFAR-10. For the non-*i.i.d.* setting,  $\alpha=0.3$  is used.

| ρ          | $1 \times 10^{-4}$ | $1 \times 10^{-3}$ | $1 \times 10^{-2}$ | $1 \times 10^{-1}$ |
|------------|--------------------|--------------------|--------------------|--------------------|
| non-i.i.d. | 87.15              | 89.71              | 89.62              | 89.46              |
| i.i.d.     | 90.64              | 92.48              | 91.97              | 92.11              |

show that FedWSQ effectively enhances the generalization, leading to more stable training and improved FL performance.

Impact on different backbone architectures Table 4 presents the performance of different backbone architectures on CIFAR-100 to evaluate the impact of FedWS and FedWSQ. For all architectures, FedWS significantly improves accuracy over the vanilla models, and FedWSQ maintains a comparable performance to FedWS with improved communication efficiency. In addition, We evaluate MobileViT [32], a vision transformer-based architecture, which also benefits from FedWS and FedWSQ. These results confirm their applicability beyond CNNs.

Comparision with FedWon We provide the comparison of FedWSQ with FedWon, which was discussed in Section 3.1. As shown in Table 1, FedWSQ substantially outperforms FedWon across all datasets and data heterogeneity settings. As illustrated in Figure 3, FedWon exhibits a much higher global minimum and a very sharp loss land-scape with a Hessian top eigenvalue of 203.4, whereas FedWSQ maintains a smoother surface with an eigenvalue of 135.8. These results demonstrate FedWSQ's superior robustness and generalization ability.

Hyper-parameters We examine the hyper-parameter  $\rho$  for WS, which handles the normalization scales. To evaluate its impact, we vary the values of  $\rho$  and assess the performance of FedWS under both non-*i.i.d.* and *i.i.d.* data distributions. As shown in Table 5, the performance of FedWS is insensitive to  $\rho$  variations. This is because during inference, normalization layers like GN cancel the effect of constant scaling, and the variance of the parameter vectors within the same convolutional layer is not significant. We select  $\rho = 1 \times 10^{-3}$ , which yields the best results.

#### 5. Conclusions

This paper presents FedWSQ, a novel FL framework using WS and the proposed DANUQ to address key FL challenges, including data heterogeneity, partial client participation, and communication bottleneck. By leveraging WS, FedWSQ enhances training stability and mitigates client drift via an implicit gradient filtering mechanism. Additionally, DANUQ efficiently compresses LMPUs while preserving essential local information, significantly reducing communication overhead even at ultra-low bit-widths. Extensive experiments on the benchmark datasets confirm that

<span id="page-7-0"></span> $<sup>^7\</sup>mbox{QLs}$  of different methods are visualized in Section C of the supplementary document.

FedWSQ consistently outperforms SOTA FL methods under various conditions. Furthermore, the mixed-precision strategies FBA and DBA effectively balance communication cost and model accuracy. These results validate the robustness of the FedWSQ framework in extreme quantization scenarios, making it highly practical for real-world FL.

# References

- <span id="page-8-7"></span>[1] Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough, and Venkatesh Saligrama. Federated learning based on dynamic regularization. In *ICLR*, 2021. [2,](#page-1-0) [6,](#page-5-1) [3](#page-2-3)
- <span id="page-8-26"></span>[2] MohammadHossein AskariHemmat, Reyhane Askari Hemmat, Alex Hoffman, Ivan Lazarevich, Ehsan Saboori, Olivier Mastropietro, Sudhakar Sah, Yvon Savaria, and Jean-Pierre David. QReg: On regularization effects of quantization. *arXiv preprint arXiv:2206.12372*, 2022. [7](#page-6-4)
- <span id="page-8-27"></span>[3] MohammadHossein AskariHemmat, Ahmadreza Jeddi, Reyhane Askari Hemmat, Ivan Lazarevich, Alexander Hoffman, Sudhakar Sah, Ehsan Saboori, Yvon Savaria, and Jean-Pierre David. QGen: On the ability to generalize in quantization aware training. *arXiv preprint arXiv:2404.11769*, 2024. [7](#page-6-4)
- <span id="page-8-0"></span>[4] Debraj Basu, Deepesh Data, Can Karakus, and Suhas N Diggavi. Qsparse-Local-SGD: Distributed SGD with quantization, sparsification, and local computations. *IEEE Journal on Selected Areas in Information Theory*, 1(1):217–226, 2020. [1](#page-0-1)
- <span id="page-8-16"></span>[5] Shengbo Chen, Cong Shen, Lanxue Zhang, and Yuanmin Tang. Dynamic aggregation for heterogeneous quantization in federated learning. *IEEE Transactions on Wireless Communications*, 20(10):6804–6819, 2021. [2,](#page-1-0) [6](#page-5-1)
- <span id="page-8-14"></span>[6] Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks. In *ICLR*, 2018. [2](#page-1-0)
- <span id="page-8-18"></span>[7] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer. 8-bit optimizers via block-wise quantization. In *ICLR*, 2022. [4,](#page-3-4) [8](#page-7-3)
- <span id="page-8-15"></span>[8] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. In *NeurIPS*, 2023. [2](#page-1-0)
- <span id="page-8-9"></span>[9] Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S. Modha. Learned Step Size Quantization. In *ICLR*, 2020. [2](#page-1-0)
- <span id="page-8-10"></span>[10] Robert M. Gray and David L. Neuhoff. Quantization. *IEEE transactions on information theory*, 44(6):2325–2383, 1998. [2](#page-1-0)
- <span id="page-8-23"></span>[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016. [6,](#page-5-1) [8,](#page-7-3) [3](#page-2-3)
- <span id="page-8-24"></span>[12] Kevin Hsieh, Amar Phanishayee, Onur Mutlu, and Phillip Gibbons. The non-IID data quagmire of decentralized machine learning. In *ICML*, 2020. [6,](#page-5-1) [3](#page-2-3)
- <span id="page-8-6"></span>[13] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical data distribution for federated visual classification. *arXiv preprint arXiv:1909.06335*, 2019. [2,](#page-1-0) [6,](#page-5-1) [3](#page-2-3)

- <span id="page-8-28"></span>[14] Forrest N Iandola, Song Han, Matthew W Moskewicz, Khalid Ashraf, William J Dally, and Kurt Keutzer. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5 MB model size. *arXiv preprint arXiv:1602.07360*, 2016. [8](#page-7-3)
- <span id="page-8-25"></span>[15] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *ICML*, 2015. [6](#page-5-1)
- <span id="page-8-11"></span>[16] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *CVPR*, pages 2704–2713, 2018. [2,](#page-1-0) [4](#page-3-4)
- <span id="page-8-12"></span>[17] Sangil Jung, Changyong Son, Seohyung Lee, Jinwoo Son, Youngjun Kwak, Jae-Joon Han, Sung Ju Hwang, and Changkyu Choi. Learning to quantize deep networks by optimizing quantization intervals with task loss. In *CVPR*, pages 4350–4359, 2019. [2](#page-1-0)
- <span id="page-8-2"></span>[18] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J Reddi, Sebastian U Stich, and Ananda Theertha Suresh. SCAFFOLD: Stochastic controlled averaging for ondevice federated learning. In *ICML*, 2020. [1](#page-0-1)
- <span id="page-8-3"></span>[19] Ahmed Khaled, Konstantin Mishchenko, and Peter Richtarik. First analysis of local GD on heterogeneous data. ´ *arXiv preprint arXiv:1909.04715*, 2019. [1](#page-0-1)
- <span id="page-8-20"></span>[20] Geeho Kim, Jinkyu Kim, and Bohyung Han. Communication-efficient federated learning with accelerated client gradient. In *CVPR*, 2024. [6,](#page-5-1) [3](#page-2-3)
- <span id="page-8-8"></span>[21] Jinkyu Kim, Geeho Kim, and Bohyung Han. Multi-level branched regularization for federated learning. In *ICML*, 2022. [2,](#page-1-0) [6,](#page-5-1) [3](#page-2-3)
- <span id="page-8-17"></span>[22] Jakub Konecnˇ y, H. Brendan McMahan, Felix X. Yu, ´ Ananda Theertha Suresh, Dave Bacon, and Peter Richtarik. ´ Federated learning: Strategies for improving communication efficiency. In *arXiv preprint*, 2017. [2](#page-1-0)
- <span id="page-8-21"></span>[23] Alex Krizhevsky. Learning multiple layers of features from tiny images. *Master's thesis, Department of Computer Science, University of Toronto*, 2009. [6](#page-5-1)
- <span id="page-8-4"></span>[24] Khiem Le, Nhan Luong-Ha, Manh Nguyen-Duc, Danh Le-Phuoc, Cuong Do, and Kok-Seng Wong. Exploring the practicality of federated learning: A survey towards the communication perspective. *arXiv preprint arXiv:2405.20431*, 2024. [1](#page-0-1)
- <span id="page-8-22"></span>[25] Ya Le and Xuan Yang. Tiny ImageNet visual recognition challenge. *CS 231N*, 7(7):3, 2015. [6](#page-5-1)
- <span id="page-8-19"></span>[26] Gihun Lee, Minchan Jeong, Yongjin Shin, Sangmin Bae, and Se-Young Yun. Preservation of the global knowledge by nottrue distillation in federated learning. In *NeurIPS*, 2022. [6,](#page-5-1) [3](#page-2-3)
- <span id="page-8-5"></span>[27] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. In *MLSys*, 2020. [2,](#page-1-0) [6,](#page-5-1) [3](#page-2-3)
- <span id="page-8-1"></span>[28] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of FedAvg on non-IID data. In *ICLR*, 2020. [1](#page-0-1)
- <span id="page-8-13"></span>[29] Darryl Lin, Sachin Talathi, and Sreekanth Annapureddy. Fixed point quantization of deep convolutional networks. In

- *International conference on machine learning*, pages 2849– 2858. PMLR, 2016. [2](#page-1-0)
- <span id="page-9-6"></span>[30] Wang Luping, Wang Wei, and Li Bo. CMFL: Mitigating communication overhead for federated learning. In *2019 IEEE 39th international conference on distributed computing systems (ICDCS)*, pages 954–964. IEEE, 2019. [1](#page-0-1)
- <span id="page-9-0"></span>[31] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communicationefficient learning of deep networks from decentralized data. In *AISTATS*, 2017. [1,](#page-0-1) [2,](#page-1-0) [6](#page-5-1)
- <span id="page-9-25"></span>[32] Sachin Mehta and Mohammad Rastegari. Mobilevit: lightweight, general-purpose, and mobile-friendly vision transformer. *arXiv preprint arXiv:2110.02178*, 2021. [8](#page-7-3)
- <span id="page-9-11"></span>[33] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization. *arXiv preprint arXiv:2106.08295*, 2021. [2,](#page-1-0) [4](#page-3-4)
- <span id="page-9-26"></span>[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. In *NeurIPS*, 2019. [3](#page-2-3)
- <span id="page-9-8"></span>[35] Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, and Alan Yuille. Micro-batch training with batch-channel normalization and weight standardization. *arXiv preprint arXiv:1903.10520*, 2019. [1,](#page-0-1) [3,](#page-2-3) [6](#page-5-1)
- <span id="page-9-9"></span>[36] Sashank J Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecnˇ y, Sanjiv Kumar, and ` Hugh Brendan McMahan. Adaptive federated optimization. In *ICLR*, 2021. [2,](#page-1-0) [6,](#page-5-1) [3](#page-2-3)
- <span id="page-9-14"></span>[37] Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani. FedPAQ: A communication-efficient federated learning method with periodic averaging and quantization. In *AISTATS*, 2020. [2,](#page-1-0) [6,](#page-5-1) [8](#page-7-3)
- <span id="page-9-20"></span>[38] Seonguk Seo, Jinkyu Kim, Geeho Kim, and Bohyung Han. Relaxed contrastive learning for federated learning. In *CVPR*, 2024. [6,](#page-5-1) [3](#page-2-3)
- <span id="page-9-19"></span>[39] Yujun Shi, Jian Liang, Wenqing Zhang, Vincent YF Tan, and Song Bai. Towards understanding and mitigating dimensional collapse in heterogeneous federated learning. In *ICLR*, 2023. [6,](#page-5-1) [3](#page-2-3)
- <span id="page-9-24"></span>[40] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In *ICLR*, 2014. [8](#page-7-3)
- <span id="page-9-1"></span>[41] Sebastian U Stich. Local SGD converges fast and communicates little. In *ICLR*, 2019. [1](#page-0-1)
- <span id="page-9-2"></span>[42] Sebastian U Stich and Sai Praneeth Karimireddy. The errorfeedback framework: Better rates for SGD with delayed gradients and compressed communication. *arXiv preprint arXiv:1909.05350*, 2019. [1](#page-0-1)
- <span id="page-9-10"></span>[43] Yan Sun, Li Shen, Shixiang Chen, Liang Ding, and Dacheng Tao. Dynamic regularized sharpness aware minimization in federated learning: Approaching global consistency and smooth landscape. In *International Conference on Machine Learning*, pages 32991–33013. PMLR, 2023. [2,](#page-1-0) [6](#page-5-1)

- <span id="page-9-3"></span>[44] Jianyu Wang and Gauri Joshi. Cooperative SGD: A unified framework for the design and analysis of local-update SGD algorithms. *Journal of Machine Learning Research*, 22(213): 1–50, 2021. [1](#page-0-1)
- <span id="page-9-16"></span>[45] Ruizhe Wang, Yeyun Gong, Xiao Liu, Guoshuai Zhao, Ziyue Yang, Baining Guo, Zhengjun Zha, and Peng Cheng. Optimizing large language model training using fp4 quantization. In *arXiv preprint arXiv:2501.17116*, 2025. [4,](#page-3-4) [8](#page-7-3)
- <span id="page-9-7"></span>[46] Jie Wen, Zhixia Zhang, Yang Lan, Zhihua Cui, Jianghui Cai, and Wensheng Zhang. A survey on federated learning: challenges and applications. *International Journal of Machine Learning and Cybernetics*, 14(2):513–535, 2023. [1](#page-0-1)
- <span id="page-9-22"></span>[47] Yuxin Wu and Kaiming He. Group normalization. In *ECCV*, 2018. [6](#page-5-1)
- <span id="page-9-21"></span>[48] Jing Xu, Sen Wang, Liwei Wang, and Andrew Chi-Chih Yao. FedCM: Federated learning with client-level momentum. *arXiv preprint arXiv:2106.10874*, 2021. [6,](#page-5-1) [3](#page-2-3)
- <span id="page-9-4"></span>[49] Haibo Yang, Minghong Fang, and Jia Liu. Achieving linear speedup with partial worker participation in non-IID federated learning. In *ICLR*, 2021. [1](#page-0-1)
- <span id="page-9-5"></span>[50] Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted SGD with faster convergence and less communication: Demystifying why model averaging works for deep learning. In *AAAI*, 2019. [1](#page-0-1)
- <span id="page-9-13"></span>[51] Dongqing Zhang, Jiaolong Yang, Dongqiangzi Ye, and Gang Hua. LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks. In *ECCV*, 2018. [2](#page-1-0)
- <span id="page-9-18"></span>[52] Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, and Chao Wu. Federated learning with label distribution skew via logits calibration. In *ICML*, 2022. [6,](#page-5-1) [3](#page-2-3)
- <span id="page-9-23"></span>[53] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. ShuffleNet: An extremely efficient convolutional neural network for mobile devices. In *CVPR*, 2018. [8](#page-7-3)
- <span id="page-9-17"></span>[54] Sijie Zhao, Tao Yue, and Xuemei Hu. Distribution-aware adaptive multi-bit quantization. In *CVPR*, 2021. [4](#page-3-4)
- <span id="page-9-12"></span>[55] Chenzhuo Zhu, Song Han, Huizi Mao, and William J. Dally. Trained ternary quantization. In *ICLR*, 2017. [2](#page-1-0)
- <span id="page-9-15"></span>[56] Weiming Zhuang and Lingjuan Lyu. FedWon: Triumphing multi-domain federated learning without normalization. In *ICLR*, 2023. [4,](#page-3-4) [6](#page-5-1)

# FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization *–Supplementary Document–*

Paper ID 9573

# <span id="page-10-0"></span>A. Technical Lemmas

<span id="page-10-1"></span>This section introduces some technical lemmas that are useful to understand our main document.

Lemma 1. *Consider any vector* v ∈ R d *. The mean subtraction of* v *is given by*

$$\bar{\mathbf{v}} = \mathbf{v} - \left(\frac{1}{d}\mathbf{1}^T\mathbf{v}\right)\mathbf{1}$$

$$= \left(\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T\right)\mathbf{v}$$

$$= (\mathbf{I} - \mathbf{P_1})\mathbf{v}$$
(11)

*where* I ∈ R d×d *is the identity matrix,* 1 ∈ R d *is a vector whose elements are all ones, and* P<sup>w</sup> *represents the projection matrix onto the vector* w*. Thus, mean subtraction is equivalent to projecting* v *onto* span{1} <sup>⊥</sup>*. In other words, this projection removes the DC (constant) component from the given vector* v*.*

Lemma 2. *Consider any vector* v¯ ∈ R <sup>d</sup> *with zero mean. Normalization of* v¯ *using its standard deviation* σ(v¯) *is given by*

$$\tilde{\mathbf{v}} = \frac{\rho}{\sigma(\bar{\mathbf{v}})} \bar{\mathbf{v}}$$

$$= \frac{\rho \sqrt{d}}{\|\bar{\mathbf{v}}\|} \bar{\mathbf{v}}.$$
(12)

*Since* v¯ *is zero-centered, its standard deviation is given by* σ(v¯) = p (v¯ <sup>T</sup> v¯)/d*.*

Lemma 3. *Consider any vector* v ∈ R d *. Let* v¯ *and* v˜ *be its mean-subtracted and standardized versions, respectively. The derivative of* v˜ *with respect to* v¯ *is then given by*

$$\frac{\partial \tilde{\mathbf{v}}}{\partial \bar{\mathbf{v}}} = \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \frac{1}{d(\sigma(\bar{\mathbf{v}}))^2} \bar{\mathbf{v}} \bar{\mathbf{v}}^T \right) 
= \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \frac{1}{\|\bar{\mathbf{v}}\|^2} \bar{\mathbf{v}} \bar{\mathbf{v}}^T \right) 
= \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \mathbf{P}_{\bar{\mathbf{v}}} \right).$$
(13)

*Also, based on Lemma [1,](#page-10-1) the derivative of* v¯ *with respect to* v *is given by*

$$\frac{\partial \bar{\mathbf{v}}}{\partial \mathbf{v}} = (\mathbf{I} - \mathbf{P_1}). \tag{14}$$

*Since* σ(v¯) = σ(v)*, by the chain rule, we can derive the gradient of a loss function* L *with respect to* v *as follows:*

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}} = \frac{\rho}{\sigma(\bar{\mathbf{v}})} \left( \mathbf{I} - \mathbf{P_1} \right) \left( \mathbf{I} - \mathbf{P_{\bar{\mathbf{v}}}} \right) \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{v}}}.$$
 (15)

# **B. Derivation of Quantization Errors**

In this section, we derive the expected quantization error, which measures the difference between the original LMPUs and their quantized values, where p(x) represents a standard normal distribution. The error is formulated as

$$\mathbb{E}\left[(\Delta w - \Delta \bar{w})^2\right] = \sum_{r=0}^{R} \int_{u_r}^{u_{r+1}} (x - q_r)^2 p(x) dx \tag{16}$$

where  $q_r$  is the quantization level. To evaluate the integral, we expand the squared term as follows:

$$\int (x - q_r)^2 p(x) dx = \frac{1}{\sqrt{2\pi}} \int (x - q_r)^2 e^{-\frac{x^2}{2}} dx$$

$$= \frac{1}{\sqrt{2\pi}} \left( \underbrace{\int x^2 e^{-\frac{x^2}{2}} dx}_{P_1} \underbrace{-2q_r \int x e^{-\frac{x^2}{2}} dx}_{P_2} + \underbrace{q_r^2 \int e^{-\frac{x^2}{2}} dx}_{P_3} \right)$$
(17)

We now calculate each term  $P_1$ ,  $P_2$ , and  $P_3$ . Let  $t = \frac{x}{\sqrt{2}}$ , which transforms  $P_1$  into

$$P_1 = 2\sqrt{2} \int t^2 e^{-t^2} dt$$

$$= -\sqrt{2}te^{-t^2} + \sqrt{2} \int e^{-t^2} dt \quad \left(\because \int u dv = uv - \int v du \quad \text{where } u = t \text{ and } dv = te^{-t^2} dt\right)$$
(18)

The definite integral over the quantization boundaries is then given by

$$-\sqrt{2}\left[te^{-t^2}\right]_{u_r/\sqrt{2}}^{u_{r+1}/\sqrt{2}} + \sqrt{2}\int_{u_r/\sqrt{2}}^{u_{r+1}/\sqrt{2}}e^{-t^2}dt = \left(u_re^{-\frac{u_r^2}{2}} - u_{r+1}e^{-\frac{u_{r+1}^2}{2}}\right) + \sqrt{\frac{\pi}{2}}\left(\operatorname{erf}\left(\frac{u_{r+1}}{\sqrt{2}}\right) - \operatorname{erf}\left(\frac{u_r}{\sqrt{2}}\right)\right)$$
(19)

Also, we can evaluate the definite integral of  $P_2$  over the qunatization boundaries as follows:

$$-2q_r \int_{u_r}^{u_{r+1}} x e^{-\frac{x^2}{2}} dx = 2q_r \left[ e^{-x^2} \right]_{u_r}^{u_{r+1}} \qquad \left( \because \int x e^{-\frac{x^2}{2}} dx = -e^{-\frac{x^2}{2}} \right)$$
$$= 2q_r \left( e^{-\frac{u_{r+1}^2}{2}} - e^{-\frac{u_r^2}{2}} \right)$$
(20)

Finally, we can easily obtain the definite integral of  $P_3$  over the qunatization boundaries by substituting  $t = \frac{x}{\sqrt{2}}$ , as follows:

<span id="page-11-0"></span>
$$q_r^2 \int_{u_r}^{u_{r+1}} e^{-\frac{x^2}{2}} dx = \sqrt{2} q_r^2 \int_{u_r/\sqrt{2}}^{u_{r+1}/\sqrt{2}} e^{-t^2} dt$$

$$= \sqrt{\frac{\pi}{2}} q_r^2 \left( \left( \text{erf} \left( \frac{u_{r+1}}{\sqrt{2}} \right) - \text{erf} \left( \frac{u_r}{\sqrt{2}} \right) \right)$$
(21)

Combining all the above derivations and unrolling the sum, we can obtain the final expression for the expected quantizaion error as follows:

$$\sum_{r=0}^{R} \int_{u_r}^{u_{r+1}} (x - q_r)^2 p(x) dx = \sum_{r=0}^{R} \left\{ \frac{1}{\sqrt{2\pi}} (2q_r - u_{r+1}) e^{-\frac{u_{r+1}^2}{2}} - \frac{1}{\sqrt{2\pi}} (2q_r - u_r) e^{-\frac{u_r^2}{2}} + \frac{1}{2} (q_r^2 + 1) \left( \operatorname{erf} \left( \frac{u_{r+1}}{\sqrt{2}} \right) - \operatorname{erf} \left( \frac{u_r}{\sqrt{2}} \right) \right) \right\} \\
= \frac{1}{2} (q_R^2 + 1) - \sqrt{\frac{2}{\pi}} q_1 e^{-\frac{u_1^2}{2}} - \frac{1}{2} q_1^2 \operatorname{erf} \left( \frac{u_1}{\sqrt{2}} \right) \\
+ \sqrt{\frac{2}{\pi}} \sum_{r=1}^{R-1} (q_r - q_{r+1}) e^{-\frac{u_{r+1}^2}{2}} + \frac{1}{2} \sum_{r=1}^{R-1} (q_r^2 - q_{r+1}^2) \operatorname{erf} \left( \frac{u_{r+1}}{\sqrt{2}} \right). \tag{22}$$

# <span id="page-12-0"></span>C. Experimental setup

Implementation details We follow most of the implementation setups and evaluation protocols in [\[1,](#page-8-7) [20,](#page-8-20) [38,](#page-9-20) [48\]](#page-9-21). The ResNet-18 architecture [\[11\]](#page-8-23) is adopted as our backbone network. Consistent with [\[12\]](#page-8-24) and common practice in FL, all BN layers in ResNet-18 are replaced with GN layers. Following the recommendation of Qiao *et al*. [\[35\]](#page-9-8), WS is applied before each GN layer. All the models are trained from scratch by using the SGD optimizer with an initial learning rate of 0.1 and a weight decay of 0.001. For the proposed model, the learning rate is exponentially decayed at each communication round by a factor of 0.995. For the other models compared, we select the learning decay parameter from {0.995, 0.998, 1} to attain the best accuracy. The global learning rate of FedAdam is set to 0.01, and that of the other methods is set to 1. Momentum is not used following the previous works [\[1,](#page-8-7) [20,](#page-8-20) [48\]](#page-9-21), and gradient clipping is applied for learning stability. Unless otherwise noted, the number of local training epochs per round is set to 5, with the batch size adjusted so that each local epoch consists of 10 iterations. The hyper-parameter of WS is set to ρ = 0.001. The source code is implemented by using the PyTorch framework [\[34\]](#page-9-26) on NVIDIA RTX 4090 GPUs. We set the number of local training epochs to 5. The batch size for local updates is adjusted so that each local epoch has 10 iterations (*i.e*., 50 iterations during a single communication round).

<span id="page-12-1"></span>Hyper-parameter selection We adopt the hyper-parameter settings of the baseline methods suggested in [\[20,](#page-8-20) [38\]](#page-9-20). Table [A](#page-12-1) summarizes the hyper-parameter settings we used, with the notations consistent with the original papers.

Table A. Summary of hyper-parameter selection

| Method         | Hyper-parameters                     |
|----------------|--------------------------------------|
| FedProx [27]   | µ = 0.001                            |
| FedAvgM [13]   | β = 0.4                              |
| FedADAM [36]   | τ = 0.001, β1<br>= 0.9, β2<br>= 0.99 |
| FedDyn [1]     | α = 0.1                              |
| FedMLB [21]    | τ = 1, λ1<br>= 1, λ2<br>= 1          |
| FedLC [52]     | τ = 1                                |
| FedNTD [26]    | τ = 1, β = 0.3                       |
| FedDecorr [39] | β = 0.01                             |
| FedRCL [38]    | τ = 0.05, β = 1, λ = 0.7             |
| FedACG [20]    | β = 0.001, λ = 0.85                  |

QLs of NUQ methods Figure [A](#page-13-0) provides a comparative visualization of the QLs adopted by different NUQ methods. The histogram illustrates the empirical distribution of LMPUs with the standard normal distribution curve. As shown in the figure, the proposed DANUQ places QLs more adaptively based on the statistical structure of LMPUs, leading to improved quantization efficiency.

<span id="page-13-0"></span>![](_page_13_Figure_1.jpeg)

Figure A. Visualization of QLs used in different NUQ methods. The histogram represents the empirical distribution of LMPUs in the 1st and 3rd ResNet-18 blocks, and the red curve denotes the standard normal distribution. The vertical dashed lines indicate the QLs chosen by different methods, NF, FP, and the proposed DANUQ, where the 4-bit representation is used.

### <span id="page-14-0"></span>D. Convergence plot evaluated on various federated learning scenarios

Figures B-D present the convergence plots of various FL methods on CIFAR-10, CIFAR-100, and Tiny-ImageNet, for i.i.d and non-i.i.d. data distributions with  $\alpha \in \{0.05, 0.1, 0.3, 0.6\}$ , using a participation rate of 5% over 100 distributed clients. As shown in the figures, FedWSQ consistently enhances the FL performance of conventional methods, outperforming those of state-of-the-art FL approaches.

<span id="page-14-1"></span>![](_page_14_Figure_2.jpeg)

Figure B. Convergence plots of our FedWS and FedWSQ compared to conventional methods on CIFAR-10 with 5% participation over 100 clients under varying Dirichlet parameters.

![](_page_15_Figure_0.jpeg)

Figure C. Convergence plots of our FedWS and FedWSQ compared to conventional methods on CIFAR-100 with 5% participation over 100 clients under varying Dirichlet parameters.

<span id="page-16-0"></span>![](_page_16_Figure_0.jpeg)

Figure D. Convergence plots of our FedWS and FedWSQ compared to conventional methods on Tiny-ImageNet with 5% participation over 100 clients under varying Dirichlet parameters.