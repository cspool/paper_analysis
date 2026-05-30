# <span id="page-5-1"></span>3.3 Training Objective

As shown in Figure 2b, we consider three trainable modules, *i.e.*, the gating model  $\phi$ , the expert models  $\{\xi_i\}_{i=0}^K$ , and the classifier  $\mu$ . We propose the following objective:

<span id="page-5-0"></span>
$$\min_{\theta} \mathcal{L}_{f} = \min_{\theta} (\mathcal{L}_{1} + \mathcal{L}_{2}), \text{ where}$$

$$\mathcal{L}_{1} = \mathbb{E}_{(\mathcal{G}, y) \sim \mathcal{D}_{s}} \mathbb{E}_{\tau^{(k)}} BCE(\phi(\tau^{(k)}(\mathcal{G})), Y(\tau^{(k)}))$$

$$\mathcal{L}_{2} = \mathbb{E}_{(\mathcal{G}, y) \sim \mathcal{D}_{s}} \mathbb{E}_{\tau^{(k)}} [CE(\mu(h(\tau^{(k)}(\mathcal{G})), y)) + \lambda \cdot d(h(\tau^{(k)}(\mathcal{G})), \xi_{0}(\mathcal{G}))]$$
(3)

- $\mathcal{L}_1$ :  $Y(\tau^{(k)}) \in \{0,1\}^{K+1}$  is the ground truth vector, and its *i*-th element is 1 if and only if  $\tau_i$  composes  $\tau^{(k)}$ . BCE is the Binary Cross Entropy. This term indicates that the gating model  $\phi$  is optimized to accurately predict a mixture of shift components.
- $\mathcal{L}_2$ : CE is the Cross Entropy function.  $d(\cdot,\cdot)$  is a distance function between two representations, and  $\lambda$  is a parameter controlling the strength of the distance penalty. In the experiments, we use the Frobenius norm as the distance function, i.e.,  $d(\mathbf{z}_1,\mathbf{z}_2) = \frac{1}{n}\|\mathbf{z}_1 \mathbf{z}_2\|_F = \frac{1}{n}\sqrt{\sum_{i=1}^n(\mathbf{z}_{1i} \mathbf{z}_{2i})^2}$ , and we use  $\lambda = 1$  for all the experiments. The second loss term optimizes the expert models and the classifier, and we prevent it from backpropagating to the gating model to avoid interference. Specifically,  $\mathcal{L}_2$  aims to improve the encoder's performance in predicting graph classes and achieves referential alignment with the reference model  $\xi_0$  via the distance function. Note that, when k > 1,  $\mathcal{L}_2$  also enforces h to be invariant to multiple shifts via the  $\tau^{(k)}$ -invariance condition.

We optimize our model via stochastic gradient descent, where  $\tau^{(k)}$  is sampled at each gradient step. Overall, GraphMETRO yields a MoE model, comprising a gating model with high predictive accuracy, expert models that are aligned and can generate invariant representations in a shared representation space, and a task-specific classifier that utilizes robust and invariant representations for class prediction.

<span id="page-6-1"></span>Table 1: Test results on the real-world datasets. We compute the p-value between the results of GraphMETRO and the state-of-the-art methods. The results of GraphMETRO is repeated five times.

|            | Node classification |              |              | Graph classification |             |  |  |
|------------|---------------------|--------------|--------------|----------------------|-------------|--|--|
|            | WebKB               | Twitch       | Twitter      | SST2                 | information |  |  |
| ERM        | 14.29 ± 3.24        | 48.95 ± 3.19 | 56.44 ± 0.45 | 80.52 ± 1.13         | No          |  |  |
| DANN       | 15.08 ± 0.37        | 48.98 ± 3.22 | 55.38 ± 2.29 | 80.53 ± 1.40         | No          |  |  |
| IRM        | 13.49 ± 0.75        | 47.21 ± 0.98 | 55.09 ± 2.17 | 80.75 ± 1.17         | Yes         |  |  |
| VREx       | 14.29 ± 3.24        | 48.99 ± 3.20 | 55.98 ± 1.92 | 80.20 ± 1.39         | Yes         |  |  |
| GroupDRO   | 17.20 ± 0.76        | 47.20 ± 0.44 | 56.65 ± 1.72 | 81.67 ± 0.45         | Yes         |  |  |
| Deep Coral | 13.76 ± 1.30        | 49.64 ± 2.44 | 55.16 ± 0.23 | 78.94 ± 1.22         | Yes         |  |  |
| SRGNN      | 13.23 ± 2.93        | 47.30 ± 1.43 | NA           | NA                   | Yes         |  |  |
| EERM       | 24.61 ± 4.86        | 51.34 ± 1.41 | NA           | NA                   | No          |  |  |
| OODGAT     | 14.41 ± 1.10        | 49.38 ± 0.87 | NA           | NA                   |             |  |  |
| DIR        | NA                  | NA           | 55.68 ± 2.21 | 81.55 ± 1.06         | No          |  |  |
| G-Mixup    | NA                  | NA           | 53.32 ± 2.75 | 77.43 ± 1.97         |             |  |  |
| GSAT       | NA                  | NA           | 56.40 ± 1.76 | 81.49 ± 0.76         | No          |  |  |
| CIGA       | NA                  | NA           | 55.70 ± 1.39 | 80.44 ± 1.24         | No          |  |  |
| GraphMETRO | 41.11 ± 7.47        | 53.50 ± 2.42 | 57.24 ± 2.56 | 81.87 ± 0.22         | No          |  |  |
| p-value    | < 0.001             | 0.023        | 0.042        | 0.081                | -           |  |  |

#### 3.4 Discussion and Analysis

Node classification tasks. While we introduce our method following a graph-level task setting, GraphMETRO is readily adaptable for node-level tasks. Instead of generating graph representations, GraphMETRO is capable of producing node-level invariant representations. Additionally, we apply stochastic transform functions to the subgraph containing a target node and identify its shift components, which is consistent with the objective in Equation [3.](#page-5-0)

Interpretability. The gating model of GraphMETRO predicts the shift components on the node or graph instance, which provides interpretations and insights into the distribution shifts in unknown datasets. In contrast, existing research on GNN generalization [\[69,](#page-13-2) [45,](#page-12-8) [6,](#page-10-7) [67\]](#page-13-3) often lacks proper identification and analysis of distribution shifts prevalent in real-world datasets. This creates a gap between human understanding of graph distribution shifts and the actual graph dynamics. To bridge this gap, we offer an in-depth study of the experiments to demonstrate GraphMETRO ' insights into the complexity of real graph distributions.

Computational cost. The forward process of f requires O(K) encoder passes, using the weighted sum aggregation from (K + 1) expert outputs. Since the extrapolation process increases the dataset size by a factor of (K + 1), the training computation complexity is O(K<sup>2</sup> |Ds|), where |Ds| is the size of the source dataset.

