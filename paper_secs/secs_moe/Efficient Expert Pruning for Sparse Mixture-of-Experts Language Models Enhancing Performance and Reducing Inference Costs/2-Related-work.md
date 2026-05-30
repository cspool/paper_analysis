# 2 Related work

Sparse Mixture-of-Experts LLMs. Shazeer et al. [\[45\]](#page-13-1) introduced the sparse MoE layer, which consists of multiple experts, each being a simple feed-forward network (FFN), and a trainable router network that selects a sparse combination of the experts to process each input. Such SMoE models can significantly increase model capacity while maintaining computational efficiency. However, this utility is ideally achieved when the router accurately and evenly assigns experts to each token during training and inference. Many works address these challenges [\[14,](#page-11-2) [28,](#page-12-5) [12,](#page-11-9) [64\]](#page-15-1). Recently, many SOTA LLMs adopt the SMoE structure to achieve high performance and computational efficiency simultaneously [\[20,](#page-11-4) [4,](#page-10-2) [50,](#page-13-3) [57\]](#page-14-2). Additionally, Zhang et al. [\[63\]](#page-14-8) propose transforming non-MoE models into SMoE models to accelerate inference, and Komatsuzaki et al. [\[25\]](#page-12-6) upcycle pretrained models by reusing the parameters to initialize SMoE models, where all experts are replicates of the original FFNs, and then fine-tune the SMoE models.

Pruning for LLMs. Pruning techniques have emerged as a crucial strategy for optimizing LLMs by reducing model size and computational costs while maintaining performance. Unstructured pruning [\[6,](#page-10-3) [15,](#page-11-5) [47,](#page-13-6) [48\]](#page-13-11) entails the removal of individual weights according to specific criteria, creating sparse networks that demand specialized hardware for efficient execution. In contrast, structured pruning [\[35,](#page-12-3) [49,](#page-13-7) [58,](#page-14-3) [18,](#page-11-6) [54,](#page-14-4) [26,](#page-12-2) [10,](#page-10-6) [59,](#page-14-9) [5\]](#page-10-7) eliminates entire structures, such as neurons or attention heads, facilitating more straightforward implementation on standard hardware. Within structured pruning, specific focus areas include attention mechanisms, where redundant heads are pruned to streamline the self-attention layers, and FFNs where unnecessary neurons are removed to enhance computational efficiency. Additionally, expert pruning for SMoE models selectively prunes the expert networks [\[34,](#page-12-4) [37,](#page-13-8) [8,](#page-10-4) [24\]](#page-12-7).

Evolutionary Strategy for Optimization. Evolutionary Strategies (ES) have been increasingly recognized for their robustness and flexibility in various optimization tasks, particularly where gradientbased methods fall short [\[55\]](#page-14-10). Notably, ES is highly effective for optimizing non-differentiable objective functions, offering a powerful alternative in scenarios where gradients are unavailable or unreliable [\[43,](#page-13-12) [22,](#page-11-10) [32,](#page-12-8) [52,](#page-14-11) [29\]](#page-12-9). Furthermore, ES excels in discrete optimization spaces, making it suitable for a wide range of combinatorial problems [\[2,](#page-10-8) [31,](#page-12-10) [30\]](#page-12-11). Recent advancements have extended the application of ES to the domain of LLMs, enabling memory-efficient fine-tuning without the need for backpropagation [\[36\]](#page-13-13).

<span id="page-3-2"></span>![](_page_3_Figure_0.jpeg)

Figure 2: We leverage EEP for two purposes: reducing the total number of experts, which lowers the memory footprint (use case 1), and reducing the number of active experts, thereby accelerating inference (use case 2).

#### 3 Background of sparse Mixture-of-Expert language model

In this section, we discuss the general concept of sparse Mixture-of-Experts (SMoE) implementation in modern decoder-only models, using the Mixtral family [20] as a specific focus. A schematic illustration is provided in Fig. 1a.

**Notations.** Let  $X \in \mathbb{R}^{n \times d}$  represent the input to a SMoE block, where n is the sequence length and d is the hidden dimension. The output of the attention block is denoted by  $Z \in \mathbb{R}^{n \times d}$ . The main parameters in the attention block are the weight matrices for computing query, key, and value:  $W_Q, W_K, W_V$ . In the SMoE structure, there are E experts, each represented by a feed-forward network (FFN) with parameters  $\theta_i$  for the i-th expert. The router network, denoted by  $W_R$ , produces routing weights  $G \in \mathbb{R}^{n \times E}$  for the sparse activation of the experts. For clarity, we omit the normalization layers and biases.

**Self-Attention Mechanism.** The self-attention mechanism computes the query, key, and value matrices as follows:  $Q = XW_Q$ ,  $K = XW_K$ ,  $V = XW_V$ . The attention scores and the output Z are then computed as:

$$\operatorname{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = \operatorname{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}, \quad \boldsymbol{Z} = \operatorname{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})\boldsymbol{W}_O, \tag{1}$$

where softmax  $(\cdot)$  denotes a row-wise softmax function. The attention mechanism produces a weighted sum of the values V, where the weights are derived from the dot product of the queries Q and keys K, scaled by the square root of key/query dimension  $\sqrt{d_k}$ . Then the weighted averaged values are mapped by the output matrix  $W_O$  to Z.

**Router Network in SMoE Structure.** The router network determines which experts to activate and how to scale their outputs. The routing weights  $G \in \mathbb{R}^{n \times E}$  are computed as:

<span id="page-3-1"></span>
$$G = \operatorname{softmax}(ZW_G). \tag{2}$$

Sparse activation of the experts is achieved by selecting the top-k routing weights for each input token. The output of the activated experts is scaled by the routing weights and aggregated to form the output of the SMoE layer H:\*

$$\forall j = 1 \dots n, \quad \boldsymbol{H}_j = \sum_{i \in \text{TopK}(\boldsymbol{G}_j)} \boldsymbol{G}_{ji} \cdot \text{FFN}_i(\boldsymbol{Z}_j), \tag{3}$$

where  $\text{TopK}(G_j)$  denotes the indices of the top-k routing weights for the j-th input token, and  $\text{FFN}_i$  denotes the function of the i-th expert, as defined below.

**FFN as Expert.** Each expert in the SMoE structure is an independent FFN with two fully-connected layers, denoted by  $W_{1i}$  and  $W_{2i}$ . When applying SwiGLU [44], an additional weight matrix  $W_{3i}$  is introduced for the activation function. The *i*-th expert processes the input as follows:

$$FFN_i(\mathbf{Z}_{sub}) = SwiGLU(\mathbf{Z}_{sub}, \mathbf{W}_{1i}, \mathbf{W}_{3i})\mathbf{W}_{2i}, \tag{4}$$

where  $Z_{sub}$  denotes the a subset of rows in Z that activates the i-th expert. Depending on the activation function, the parameters of the i-th expert are either  $\theta_i = \{W_{1i}, W_{2i}\}$  or  $\theta_i = \{W_{1i}, W_{2i}, W_{3i}\}$ .

<span id="page-3-0"></span><sup>\*</sup>The top-k routing weights may be further normalized to sum to 1; this nuance is omitted here.

#### 4 Method

In this section, we introduce our proposed approach for optimizing SMoE LLMs through expert pruning and merging. We aim to enhance the efficiency and performance of SMoE architectures by leveraging evolutionary strategies. Our method addresses the challenges of large and complex search spaces without incurring the prohibitive computational costs associated with gradient-based optimization. The subsequent subsections elaborate on our motivation (Sec. 4.1), the configuration of the parameter space (Sec. 4.2), the evolutionary optimization strategy employed to achieve our objectives (Sec. 4.3), and the use cases we apply EEP (Sec. 4.4).

