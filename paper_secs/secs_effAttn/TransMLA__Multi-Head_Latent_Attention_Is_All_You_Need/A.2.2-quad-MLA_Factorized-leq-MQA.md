# $A.2.2 \quad MLA_{Factorized} \leq MQA$

Consider an MLA-Factorized model where queries are  $\mathbf{q}_{t,i} = W_i^Q \mathbf{x}_t$  (assuming  $W_i^Q \in \mathbb{R}^{d \times D}$  is the i-th block of  $W^Q$ ) and keys are  $\mathbf{k}_{j,i} = (W_i^{UK}(W^{DKV}\mathbf{x}_j))$ . The attention score for head i involves  $\mathbf{q}_{t,i}^{\top}\mathbf{k}_{j,i}$ :

$$\mathbf{q}_{t,i}^{\top} \mathbf{k}_{j,i} = (W_i^Q \mathbf{x}_t)^{\top} (W_i^{UK} (W^{DKV} \mathbf{x}_j)). \tag{25}$$

This can be rewritten as:

$$\mathbf{q}_{t,i}^{\top} \mathbf{k}_{j,i} = (\underbrace{(W_i^{UK})^{\top} W_i^Q}_{W_i^{Q}} \mathbf{x}_t)^{\top} (W^{DKV} \mathbf{x}_j).$$
 (26)

Let  $\hat{\mathbf{q}}_{t,i} = W_i^{\prime Q} \mathbf{x}_t \in \mathbb{R}^{2gd}$  and  $\mathbf{c}_i^{KV} = W^{DKV} \mathbf{x}_j \in \mathbb{R}^{2gd}$ . The computation of attention output

$$\mathbf{o}_{t,i} = \sum_{j} \operatorname{softmax}_{j} \left( \frac{\hat{\mathbf{q}}_{t,i}^{\mathsf{T}} \mathbf{c}_{j}^{KV}}{\sqrt{d}} \right) W_{i}^{UV} \mathbf{c}_{j}^{KV}, \tag{27}$$

$$\mathbf{y}_t = W^O[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,h}]$$

$$\mathbf{y}_{t} = W^{O}[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,h}]$$

$$= W^{O}\begin{bmatrix} W_{1}^{UV} & & \\ & W_{2}^{UV} & \\ & & \ddots & \\ & & & W_{h}^{UV} \end{bmatrix} \begin{pmatrix} \operatorname{softmax}_{j}(\frac{\hat{\mathbf{q}}_{t,1}^{\top}\mathbf{c}_{j}^{KV}}{\sqrt{d}})\mathbf{c}_{j}^{KV} \\ & \vdots \\ \operatorname{softmax}_{j}(\frac{\hat{\mathbf{q}}_{t,1}^{\top}\mathbf{c}_{j}^{KV}}{\sqrt{d}})\mathbf{c}_{j}^{KV} \end{pmatrix} . \tag{28}$$

This is an MQA formulation where each modified query  $\hat{\mathbf{q}}_{t,i}$  (now of dimension 2gd) attends to a shared key and value  $\mathbf{c}_i^{KV}$ . This indicates that the computations within MLA-Factorized can be structured to use shared intermediate key and value representations akin to MQA's core. Thus, any MLA-Factorized model can be represented as an MQA model with a shared key/value of dimension 2gd.

