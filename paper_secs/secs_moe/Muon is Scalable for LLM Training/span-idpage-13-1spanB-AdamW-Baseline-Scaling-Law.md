# <span id="page-13-1"></span>**B** AdamW Baseline Scaling Law

To ensure the fairness and accuracy of our experiments, we conducted a series of experiments on our proprietary dataset to derive scaling law parameters that are optimal for AdamW. This includes determining the optimal model  $\operatorname{size}(N)$ , number of training  $\operatorname{tokens}(D)$ , learning  $\operatorname{rate}(\eta)$ , batch  $\operatorname{size}(B)$  under a constrained computational budget (FLOPs, C). (Kaplan et al. 2020; Hoffmann et al. 2022; Bi et al. 2024) Table 9 presents the results of our systematic parameter search process.

<span id="page-13-3"></span>Table 9: Empirical Relationships Between Scaling Law Parameters and Computational Budget (FLOPs)

| N(C)                            | D(C)                            | $ $ $\eta(C)$                    | B(C)                            |
|---------------------------------|---------------------------------|----------------------------------|---------------------------------|
| $0.0483359 \cdot C^{0.5112684}$ | $3.4480927 \cdot C^{0.4887316}$ | $0.0127339 \cdot C^{-0.0574752}$ | $0.0065202 \cdot C^{0.4137915}$ |

<span id="page-14-2"></span>Figure 5: Optimization Landscapes for Scaling Law Hyper-parameters Across FLOPs Budgets

**Hyper-Parameters Search** To systematically identify optimal scaling law hyper-parameters in the AdamW baseline, we adopted a multistage search protocol. First, we selected multiple computational budgets (FLOPs levels) and initialized model sizes, learning rates, and batch sizes based on empirical guidelines from prior studies. For each fixed FLOPs constraint, we varied the model size N while adjusting the training token count D inversely to maintain C=6ND, thereby exploring the trade-off between model capacity and data efficiency. Each configuration was trained to convergence, and the validation loss was recorded to determine the Pareto-optimal combinations of N and D. Subsequently, with the optimal N-D pairs fixed, we refined the learning rate and batch size through grid searches, ensuring stability and convergence across configurations. To mitigate local minima and enhance robustness, this iterative procedure was repeated 2-3 times, progressively narrowing the hyper-parameter space.

The optimization process is further illustrated in Figure 5, which depicts the loss landscapes as functions of training tokens, learning rate, and batch size across varying FLOPs budgets. Each bowl-shaped curve represents the loss surface for a specific FLOPs level, with a distinct global minimum corresponding to the optimal hyper-parameter configuration.

### <span id="page-14-0"></span>**C** Model Architecture

Muon is agnostic to model architectures, and we used a model similar to Deepseek-V3-Small as described in DeepSeek-AI et al. 2024, because it is a strong model with open weights as a baseline. We made several small modifications in the Moonlight model and listed them here:

**Multi-token Prediction (MTP)** MTP has not shown significant benefits to pretraining in our experiments. For simplicity, we do not introduce MTP layers into the Moonlight model.

**Auxfree Bias Update** In DeepSeek-AI et al. 2024, auxfree bias is updated by:  $b_i = b_i + u \times \text{sign}(e_i)$ , where u is the update ratio,  $b_i$  is the bias for the ith expert, and  $e_i$  is the expert's violating ratio. We slightly modified the update rule as:  $b_i = b_i + u \times (\text{sign}(e_i) - \text{sign}(e).\text{mean}())$ , where sign(e).mean() is the average of the signs of all expert's violating ratio, in order to control the magnitude of the bias, while does not change the topk selection logic.

**Gate Scaling Factor** Deepseek-V2-Lite did not use the gate scaling factor, and Deepseek-V3 used a scaling factor of 2.5. We used a scaling factor of 2.446 to control a similar output rms like dense models. The code for calculating our gate scaling factor can be found in Figure 6.

