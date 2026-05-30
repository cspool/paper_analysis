# A.2 GraphMETRO's Approach

GraphMETRO addresses these limitations through its novel architecture and training objective:

1. Decomposition of shifts: Instead of learning a single invariant predictor, GraphMETRO decomposes the complex shift into K shift components:

$$E = (E_1, E_2, ..., E_K) (5)$$

where each E<sup>i</sup> represents a specific type of graph transformation.

2. Mixture-of-Experts: The gating model ϕ and expert models {ξi} K <sup>i</sup>=0 allow for adaptive handling of heterogeneous shifts:

$$h(\mathcal{G}) = \sum_{i=0}^{K} w_i(\mathcal{G}) \cdot \xi_i(\mathcal{G})$$
(6)

where wi(G) = ϕ(G)<sup>i</sup> are instance-dependent weights.

3. Referential alignment: The training objective enforces alignment between expert outputs and a reference model:

$$\mathcal{L}_2 = \mathbb{E}_{(\mathcal{G}, y) \sim \mathcal{D}_s} \mathbb{E}_{\tau^{(k)}} [CE(\mu(h(\tau^{(k)}(\mathcal{G})), y)) + \lambda \cdot d(h(\tau^{(k)}(\mathcal{G})), \xi_0(\mathcal{G}))]$$
(7)

### A.3 Theoretical Guarantees

We now provide theoretical guarantees for GraphMETRO's performance:

Theorem 1 (Shift-Invariance) *For any graph* G *and shift component* τ<sup>i</sup> *, the encoder* h *satisfies:*

$$h(\tau_i(\mathcal{G})) = h(\mathcal{G}) \tag{8}$$

Proof 1 *Given the gating model's ability to identify shift components and the expert models' invariance properties:*

$$h(\tau_i(\mathcal{G})) = \sum_{j=0}^{K} w_j(\tau_i(\mathcal{G})) \cdot \xi_j(\tau_i(\mathcal{G}))$$
(9)

$$= w_i(\tau_i(\mathcal{G})) \cdot \xi_i(\tau_i(\mathcal{G})) \tag{10}$$

$$= w_i(\mathcal{G}) \cdot \xi_0(\mathcal{G}) \tag{11}$$

$$=h(\mathcal{G})\tag{12}$$

*where the second equality holds because the gating model identifies* τ<sup>i</sup> *, and the third equality follows from the definition of referential invariant representation.*

This theorem guarantees that GraphMETRO can handle individual shift components. We can extend this to combinations of shifts:

Theorem 2 (Composition of Shifts) *For any graph* G *and combination of* k *shift components* τ (k) = τi<sup>1</sup> ◦ τi<sup>2</sup> ◦ ... ◦ τi<sup>k</sup> *, the encoder* h *approximately satisfies:*

$$h(\tau^{(k)}(\mathcal{G})) \approx h(\mathcal{G})$$
 (13)

Proof 2 *We prove this theorem by induction on* k*, the number of shift components.*

*Base case (*k = 1*): This is directly given by Theorem 1.*

*Inductive step: Assume the theorem holds for* k − 1 *shift components. We need to prove it holds for* k *shift components.*

Let 
$$\tau^{(k)} = \tau_{i_k} \circ \tau^{(k-1)}$$
 where  $\tau^{(k-1)} = \tau_{i_1} \circ \tau_{i_2} \circ \dots \circ \tau_{i_{k-1}}$ .

$$h(\tau^{(k)}(\mathcal{G})) = h(\tau_{i_k}(\tau^{(k-1)}(\mathcal{G})))$$

$$\tag{14}$$

$$\approx h(\tau^{(k-1)}(\mathcal{G}))$$
 (by Theorem 1) (15)

$$\approx h(\mathcal{G})$$
 (by induction hypothesis) (16)

*The approximation in the second line comes from the fact that the gating model* ϕ *may not perfectly identify the shift component* τ<sup>i</sup><sup>k</sup> *when applied after* τ (k−1)*. However, our training objective* L<sup>2</sup> *explicitly minimizes:*

$$\mathbb{E}_{\tau^{(k)}}[d(h(\tau^{(k)}(\mathcal{G})), \xi_0(\mathcal{G}))] \tag{17}$$

*This ensures that even for compositions of shifts, the output of* h *remains close to the reference model* ξ0*, which is invariant to all shifts.*

*Therefore, by induction, the theorem holds for any* k ≥ 1*.*

Theorem 3 (Generalization Bound) *Let* L(·, ·) *be the cross-entropy loss. For any distribution* D<sup>t</sup> *resulting from a combination of shift components in* τ (k) *, the generalization error of GraphMETRO satisfies:*

$$\mathbb{E}_{(\mathcal{G},y)\sim\mathcal{D}_t}[\mathcal{L}(f(\mathcal{G}),y)] \le \mathbb{E}_{(\mathcal{G},y)\sim\mathcal{D}_s}[\mathbb{E}_{\tau^{(k)}}[\mathcal{L}(f(\tau^{(k)}(\mathcal{G})),y)]] + \epsilon \tag{18}$$

*where* ϵ *is a small constant depending on the complexity of the model and the number of samples.*

**Proof 3** 1) Our training objective minimizes:

$$\mathcal{L}_{train} = \mathbb{E}_{(\mathcal{G}, y) \sim \mathcal{D}_s} [\mathbb{E}_{\tau^{(k)}} [\mathcal{L}(f(\tau^{(k)}(\mathcal{G})), y)]]$$
(19)

2) Recall that  $f = \mu \circ h$ , where  $\mu$  is implemented as a single linear layer with a softmax output, and  $\mathcal{L}$  is the cross-entropy loss. This combination is stictly convex with respect to the inputs to  $\mu$  if they are not perfectly collinear. Therefore, by Jensen's inequality:

$$\mathbb{E}_{\tau^{(k)}}[\mathcal{L}(f(\tau^{(k)}(\mathcal{G})), y)] > \mathcal{L}(\mu(\mathbb{E}_{\tau^{(k)}}[h(\tau^{(k)}(\mathcal{G}))]), y) \tag{20}$$

3) This implies:

<span id="page-17-1"></span>
$$\mathcal{L}_{train} > \mathbb{E}_{(\mathcal{G}, y) \sim \mathcal{D}_s} [\mathcal{L}(\mu(\mathbb{E}_{\tau^{(k)}}[h(\tau^{(k)}(\mathcal{G}))]), y)], \tag{21}$$

hence, minimizing  $\mathcal{L}_{train}$  also minimizes the left hand side of the inequality.

4) Now, consider any target distribution  $\mathcal{D}_t$  resulting from a combination of shift components in  $\tau^{(k)}$ . By definition, we can express  $\mathcal{D}_t$  as:

$$\mathcal{D}_t = \{ \tau^{(k)}(\mathcal{G}) : \mathcal{G} \sim \mathcal{D}_s, \tau^{(k)} \sim P(\tau^{(k)}) \}$$
(22)

where  $P(\tau^{(k)})$  is some distribution over the possible combinations of shift components.

5) Therefore:

<span id="page-17-2"></span>
$$\mathbb{E}_{(\mathcal{G},y)\sim\mathcal{D}_{t}}[\mathcal{L}(f(\mathcal{G}),y)] = \mathbb{E}_{(\mathcal{G},y)\sim\mathcal{D}_{s}}[\mathbb{E}_{\tau^{(k)}\sim P(\tau^{(k)})}[\mathcal{L}(f(\tau^{(k)}(\mathcal{G})),y)]]$$

$$\leq \mathbb{E}_{(\mathcal{G},y)\sim\mathcal{D}_{s}}[\mathbb{E}_{\tau^{(k)}}[\mathcal{L}(f(\tau^{(k)}(\mathcal{G})),y)]]$$

$$= \mathcal{L}_{train}$$
(23)

The inequality in the second line holds because our training objective considers a wider range of transformations than those in the actual target distribution.

- 6) Equations (21) and (23) show that minimizing  $\mathcal{L}_{train}$  implies both finding a model with low true risk and a model that is more invariant to  $\tau^{(k)}(\mathcal{G})$ , since Equation (21) shows the loss is lower if  $\forall \tau \in supp(\tau^{(k)}), h(\tau(\mathcal{G})) = \mathbb{E}_{\tau^{(k)}}[h(\tau^{(k)}(\mathcal{G}))].$
- 6) The gap between the true risk and the empirical risk can be bounded by a constant  $\epsilon$  that depends on the complexity of the model and the number of samples, according to standard statistical learning theory. Therefore, we get:

$$\mathbb{E}_{(\mathcal{G},y)\sim\mathcal{D}_t}[\mathcal{L}(f(\mathcal{G}),y)] \leq \mathbb{E}_{(\mathcal{G},y)\sim\mathcal{D}_s}[\mathbb{E}_{\tau^{(k)}}[\mathcal{L}(f(\tau^{(k)}(\mathcal{G})),y)]] + \epsilon \tag{24}$$

These theoretical results demonstrate that GraphMETRO can effectively handle complex, heterogeneous graph distribution shifts by:

- Decomposing shifts into manageable components.
- Adaptively combining expert models to handle instance-specific shifts.
- Ensuring invariance to individual and combined shift components.
- Providing a tractable upper bound on the generalization error for shifted distributions.

Compared to existing approaches that struggle with vast environment spaces or heterogeneous shifts, GraphMETRO's adaptive mixture-of-experts architecture and alignment-based training objective provide a more flexible and scalable solution for real-world graph distribution shifts.

### <span id="page-17-0"></span>**B** Experimental Details

**Experimental settings on synthetic datasets.** We randomly split each dataset into training (80%), validation (20%), and testing (20%) subsets. We consider transformations for k=2, i.e.,  $\tau^{(2)}$ , which includes both single transformations and compositions of two different transformation functions. For the compositions, we exclude trivial combinations (e.g., adding and dropping edges) and combinations that may result in an empty graph (e.g., random subgraph sampling and node dropping). These transformations are applied to the testing datasets to create multiple variants for testing environments.

**Model architecture and optimization**. We summarize the model architecture and hyperparameters for synthetic experiments (Section 4.2) in Table 2. We use the Adam optimizer with weight decay set to 0. The encoder (backbone) architecture, including the number of layers and hidden dimensions, is selected based on validation performance from the ERM model and then fixed for each encoder during GraphMETRO training.

<span id="page-18-0"></span>

|                          | Node classification                 |          | Graph c         | lassification   |  |  |  |
|--------------------------|-------------------------------------|----------|-----------------|-----------------|--|--|--|
|                          | DBLP                                | CiteSeer | IMDB-MULTI      | REDDIT-BINARY   |  |  |  |
| Backbone                 | Graph Attention Networks (GAT) [63] |          |                 |                 |  |  |  |
| Activation               | PeLU                                |          |                 |                 |  |  |  |
| Dropout                  | 0.0                                 |          |                 |                 |  |  |  |
| Number of layers         | 3                                   | 3        | 2               | 2               |  |  |  |
| Hidden dimension         | 64                                  | 32       | 128             | 128             |  |  |  |
| Global pool              | NA                                  | NA       | global add pool | global add pool |  |  |  |
| Epoch                    | 100                                 | 200      | 100             | 100             |  |  |  |
| Batch size               | NA                                  | NA       | 32              | 32              |  |  |  |
| ERM Learning rate        | 1e-3                                | 1e-3     | 1e-4            | 1e-3            |  |  |  |
| GraphMETRO Learning rate | 1e-3                                | 1e-3     | 1e-4            | 1e-3            |  |  |  |

Table 2: Architecture and hyperparameters on synthetic experiments.

For the real-world datasets, we use the same encoder and classifier from the implementation of the GOOD benchmark<sup>4</sup>. The results for the baseline methods, except for Twitter (recently added to the benchmark), are reported by the GOOD benchmark. We summarize the architecture and hyperparameters used for real-world experiments below.

|                                                             | Node classification WebKB Twitch |                           | Graph cla                                              | ssification               |  |  |
|-------------------------------------------------------------|----------------------------------|---------------------------|--------------------------------------------------------|---------------------------|--|--|
|                                                             |                                  |                           | Twitter                                                | SST2                      |  |  |
| Backbone                                                    | Graph Convolutional Network [25] |                           | Graph Isomorphism Network [70]<br>w/ Virtual node [18] |                           |  |  |
| Activation                                                  | ReLU                             |                           |                                                        |                           |  |  |
| Dropout                                                     | 0.5                              |                           |                                                        |                           |  |  |
| Number of layers                                            | 3                                |                           |                                                        |                           |  |  |
| Hidden dimension<br>Global pool                             | NA                               | NA                        | 300<br>global mean pool                                | global mean pool          |  |  |
| Epoch Batch size ERM Learning rate GraphMETRO Learning rate | 100<br>NA<br>1e-3<br>1e-2        | 100<br>NA<br>1e-3<br>1e-2 | 200<br>32<br>1e-3<br>1e-3                              | 200<br>32<br>1e-3<br>1e-3 |  |  |

Table 3: Architecture and hyperparameters on real-world datasets.

For all datasets, we conduct a grid search for GraphMETRO learning rates due to the difference in architecture compared to traditional GNN models. GraphMETRO uses multiple GNN encoders, serving as expert modules.

#### <span id="page-18-3"></span>C Stochastic Transform Functions

We built a library of 11 stochastic transform functions on top of PyG<sup>5</sup>, and used 5 of them in our synthetic experiments for demonstration purposes. Each function allows for one or more hyperparameters to control the degree of the transformation, such as the probability parameter in a Bernoulli distribution for dropping edges. A certain amount of randomness is retained in each stochastic transform function, ensuring diversity in the generated graphs.

<span id="page-18-1"></span><sup>4</sup>https://github.com/divelab/GOOD/tree/GOODv1

<span id="page-18-2"></span><sup>&</sup>lt;sup>5</sup>https://github.com/pyg-team/pytorch\_geometric

```
stochastic_transform_dict = {
    'mask_edge_feat': MaskEdgeFeat(p, fill_value),
    'noisy_edge_feat': NoisyEdgeFeat(p),
    'edge_feat_shift': EdgeFeatShift(p),
    'mask_node_feat': MaskNodeFeat(p, fill_value),
    'noisy_node_feat': NoisyNodeFeat(p),
    'node_feat_shift': NodeFeatShift(p),
    'add_edge': AddEdge(p),
    'drop_edge': DropEdge(p),
    'drop_node': DropNode(p),
    'drop_path': DropPath(p),
    'random_subgraph': RandomSubgraph(k)
}
```

We observed that different sets or numbers of transform functions can impact model performance. Specifically, we use stochastic transform functions as the foundation for the decomposed target distribution shifts. Ideally, these functions should be diverse and cover different potential aspects of distribution shifts. However, using a large number of transform functions increases the demand on the gating model's expressiveness, as it must distinguish between different transformed graphs. Additionally, more transform functions increase computational cost due to the larger number of experts. An ablation study in Appendix [D.3](#page-20-2) further validates this analysis.

In practice, the stochastic transform functions proved effective on real-world datasets, suggesting their ability to represent various distribution shifts. Exploring common base transform functions to better capture real-world distribution shifts would be an interesting direction for future research.

