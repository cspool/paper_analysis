# <span id="page-4-0"></span>3.4 Training with Long Context Distillation

Within the knowledge distillation framework, elements like hidden representations (Jiao et al., 2020), attention dependencies (Wang et al., 2020), and relationships among representations (Park et al., 2021) are considered essential for effective knowledge transfer. In this paper, we introduce long context distillation (LCD) as the contextualized knowledge that primarily guides the student model. Specifically, the teacher model, FiD (Izacard and Grave, 2021), which processes longer contextual inputs, theoretically contains more information due to its richer context. This enables it to activate more specific internal knowledge, serving as a supervisory model. The teacher model aids the student model, T5 (Roberts et al., 2020a), which is of the same size but uses shorter contextual inputs, in activating richer feature representations and knowledge. The optimization objective for the student model at each mini-batch  $z_r = (x_r, y_r)$  is:

$$\mathcal{L}_{s}(\theta_{s}, \theta_{t}, z_{r}) = \alpha \mathcal{L}_{ce}(y_{r}, S(x_{r}; \theta_{s})) + (1 - \alpha) \mathcal{L}_{ce}(T(x_{r}; \theta_{t}), S(x_{r}; \theta_{s}))$$
(5)

where we have a teacher model denoted as  $T(\cdot; \theta_t)$  and a student model denoted as  $S(\cdot; \theta_s)$ . The corresponding model parameters are  $\theta_t$  and  $\theta_s$ .

As illustrated on the right of Figure 2, we perform additional representation alignment to facilitate better knowledge transfer. In our distillation process, both the teacher and student models have L layers. The input text is processed through these layers, yielding corresponding output hidden states  $\{H_l^t\}_{l=0}^L$  and  $\{H_l^s\}_{l=0}^L$ , along with attention matrices  $\{A_l^t\}_{l=1}^L$  and  $\{A_l^s\}_{l=1}^L$ . For aligning hidden states, we calculate the proximity between the teacher's and student's hidden states using cosine distance (COS) (Park et al., 2021).

$$\mathcal{L}_{\text{hid}} = -\operatorname{COS}(H_l^s, H_l^t) \tag{6}$$

While for aligning attention dependencies, we follow (Jiao et al., 2020) to optimize the mean square

<span id="page-4-1"></span><sup>&</sup>lt;sup>3</sup>We conduct a detailed analysis of the reasons behind the hypernetwork in the A.3.

error (MSE) between the attention matrices of the teacher and the student:

$$\mathcal{L}_{\text{attn}} = -\operatorname{MSE}(A_l^s, A_l^t) \tag{7}$$

The overall objective for knowledge transfer is:

$$\mathcal{L}_{\text{align}}(H_l^s, H_l^t, A_l^s, A_l^t) = \mathcal{L}_{\text{attn}} + \mathcal{L}_{\text{hid}} \quad (8)$$

The overall objective for training AAG is the weighted sum of the two objectives:

$$\mathcal{L} = \mathcal{L}_s + \lambda \mathcal{L}_{align} \tag{9}$$

