# <span id="page-4-0"></span>5.1 Training objective with Expert-Redundancy Regularization

Expert-Redundancy Regularization. Pre-trained base model as the backbone network contains extensive general knowledge derived from large-scale training data, providing a strong foundation for various tasks. However, in MoA, the specialization of adapters is often hindered by learning redundant information overlapping with the backbone's capabilities, leading to the redundancy shown in Section 3.

We introduce a regularization strategy to address this limitation by encouraging experts to learn specialized skills while the backbone handles common patterns in the data. This synergy between the backbone and experts improves the overall effectiveness of the model. Specifically, we train embedding for each layer of the backbone, denoted as  $\mathbf{c} = [\mathbf{c}_1, \dots, \mathbf{c}_L] \in \mathbb{R}^{L \times d}$ . The fitness of the backbone at layer-l for token  $\mathbf{x}_i$  is computed as  $\langle \mathbf{x}_i, \mathbf{c}_l \rangle$ . To compare the backbone layer with the N adapters, we compute its relative fitness for input token  $\mathbf{x}_i$  at layer-l as

<span id="page-4-5"></span>
$$\mathbf{v}_{l,i} = \frac{\exp\left\langle \mathbf{x}_i, \mathbf{c}_l \right\rangle}{\exp\left\langle \mathbf{x}_i, \mathbf{c}_l \right\rangle + \sum_{n \in A_l} \exp\left\langle \mathbf{x}_i, \mathbf{e}_n \right\rangle} \quad (6)$$

The relative fitness of backbone layer-l for the whole input sequence can be computed by averaging  $\mathbf{v}_{l,i}$  over all the s tokens, i.e.,  $\frac{1}{s}\sum_{i=1}^{s}\mathbf{v}_{l,i}$ , which is then used to weight the merged adapters  $\Delta \mathbf{y}$  in Eq. (1), i.e.,

<span id="page-4-6"></span>
$$\Delta \mathbf{y} \leftarrow \left(1 - \frac{1}{s} \sum_{i=1}^{s} \mathbf{v}_{l,i}\right) \sum_{n \in A_l} \mathbf{u}_n \mathbf{B}_n \mathbf{A}_n \mathbf{x}$$
 (7)

To encourage the complementarity of adapters to the backbone base model, we apply a regularization

$$R(\mathbf{e}_{1:N}, \mathbf{c}_{1:L}) = \frac{1}{Ls} \sum_{l=1}^{L} \sum_{i=1}^{s} \mathbf{v}_{l,i}$$
 (8)

which encourages the usage of backbone layers and thus steers the adapters' focus on learning complementary knowledge and skills. Our empirical results in Section 6.2 demonstrate the effectiveness of  $R(\mathbf{e}_{1:N}, \mathbf{c}_{1:L})$  on encouraging backbone utilization and promoting diversity among experts. Overall training objective of SMOA is defined as

<span id="page-4-2"></span>
$$\min_{\theta_{1:N}, \mathbf{e}_{1:N}, \mathbf{c}_{1:L}} \quad \mathbb{E}_{(\mathbf{x}, \mathbf{y})} \mathcal{L}[F(\mathbf{x}; \theta_{1:N}, \mathbf{e}_{1:N}, \mathbf{c}_{1:L}), \mathbf{y}] \\ -\alpha R(\mathbf{e}_{1:N}, \mathbf{c}_{1:L}),$$

where  $F(\mathbf{x}; \theta_{1:N}, \mathbf{e}_{1:N}, \mathbf{c}_{1:L})$  denotes the SMoA model output for input  $\mathbf{x}$  (each layer output follows Eq. (1)), and  $\mathcal{L}(\cdot, \cdot)$  denotes the loss of the target task. The weight  $\alpha$  controls the trade-off between performance and diversity.

#### <span id="page-4-1"></span>5.2 Training Algorithm

The full training procedure for SMoA is outlined in Algorithm 1, which dynamically optimizes the cross-layer shared adapter pool  $\theta_{1:N}$ , the global router (parameterized by  $\mathbf{e}_{1:N}$ ), and the weight of merged adapters (parameterized by  $\mathbf{c}_{1:L}$ ). At each layer-l, it selects a sparse subset of experts  $A_l$  most related to the input tokens (in terms of their routing scores  $\mathbf{w}_{n,i}$ ). It then merges the selected adapters with the recomputed routing weights  $\mathbf{u}_n$  and the

backbone-adapter balancing weight (based on  $\mathbf{v}_{l,i}$ ). The merged model is trained to minimize the regularized loss in Eq. (9). This iterative training encourages expert specialization while maintaining synergy with the backbone, hence resulting in MoA with less redundancy and better generalization performance to diverse tasks.

### **Algorithm 1** SMOA Training

- 1: **Input:** L-layer pre-trained model, N experts
- 2: **Initialize:** experts  $\theta_{1:N}$ , expert embedding  $\mathbf{e}_{1:N}$ , backbone embedding  $\mathbf{c}_{1:L}$
- 3: **for** each training step **do**
- 4: **for**  $l \in \{1, \dots, L\}$  **do**
- 5:  $\mathbf{w}_{n,i} \leftarrow \text{calculate token-to-expert routing}$ score by Eq. (3) and (5)
- 6:  $A_l \leftarrow$  select adapters by solving Eq. (4), to select  $n_l$  experts
- 7:  $\mathbf{u}_n \leftarrow \text{recompute routing weights for selected experts by Eq. (3) and (5)}$
- 8:  $\mathbf{v}_{l,i} \leftarrow \text{calculate the fitness of the backbone layer by Eq. (6)}$
- 9:  $\Delta y \leftarrow$  merge selected adapters by Eq. (7)
- 10: end for
- 11:  $\theta_{1:N}, \mathbf{e}_{1:N}, \mathbf{c}_{1:L} \leftarrow \text{solving Eq. (9)}$
- <span id="page-5-1"></span>12: end for
- 13: **Return:** Optimized  $\theta_{1:N}$ ,  $\mathbf{e}_{1:N}$ ,  $\mathbf{c}_{1:L}$

