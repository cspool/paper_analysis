# 4 Training Techniques

#### 4.1 Gating Logit Normalization

One phenomenon that we have frequently observed during the training of MoE models is that its gating layers sometimes tend to yield distributions with high entropy, i.e., the topk probabilities for the selected experts are only marginally greater than those for the nonselected experts. Consequently, the output of the MoE layer is approximated as follows:

$$y_i \approx \frac{1}{k} \sum_{j \in \mathcal{E}_i} \text{Expert}_j(x_i),$$

In this scenario, the output is effectively a simple average of the selected expert outputs, rather than a weighted average. This suggests a uniformity among experts, indicating that the gating mechanism fails to discriminate effectively between different experts, which can be detrimental to model performance.

<span id="page-4-0"></span>![](_page_4_Figure_8.jpeg)

Figure 2: A comparison of gate distribution with and without logit normalization. The black dashed line corresponds to the baseline of uniform probability 1/16.

Although the underlying cause of this phenomenon still warrants further investigation, we have identified a straightforward solution. This remedy involves introducing a normalization step prior to the softmax function in the gating layer to ensure a more distinct gate output distribution. Specifically, we propose modifying the gating layer [\(1\)](#page-1-1) as follows:

$$z = Wx + b$$

$$\tilde{z} = \lambda \cdot \frac{z - \mu}{\sigma}$$

$$g = \operatorname{softmax}(\tilde{z}),$$
(6)

In this revised formulation, the vector z is first normalized by subtracting its mean µ and dividing by its standard deviation σ. It is then scaled by a hyper-parameter λ, resulting in a transformed vector z˜ with zero mean and a standard deviation controlled by λ. This adjustment ensures that the output vector z˜ is suitably scaled before applying the softmax function. The parameter λ plays the important role of determining the sharpness of the softmax output distribution. Specifically, a higher value of λ leads to a sharper, more focused distribution. This sharper gating mechanism is intended to enhance the model's ability to effectively differentiate between the contributions of various experts, thereby potentially improving the overall performance of the MoE model.

To validate our proposed methodology, we conducted a small-scale experiment using an MoE model equipped with 2.5 billion parameters and 16 experts. We compared models trained both with and without gating logit normalization and varied the hyperparameter λ. The results are illustrated in Figure [2](#page-4-0) and Figure [3.](#page-6-0) In Figure [2](#page-4-0) we show the output distribution of a gate for a model trained with gating logit normalization is significantly sharper than the one trained without. In the upper plots of Figure [3,](#page-6-0) we can see that all models trained with gating logit normalization exhibit significantly lower training losses and token drop rates compared to that without normalization. Additionally, we analyzed the ratios of M ax1/M ax<sup>2</sup> and M ax2/M ax3, where M ax<sup>i</sup> represents the i-th largest probability in the gate output distribution. These ratios are important indicators of the discriminative power of the expert router. A higher M ax1/M ax<sup>2</sup> and M ax2/M ax<sup>3</sup> ratio suggests a more effective differentiation among

experts. As shown in the lower plots of Figure 3, increasing  $\lambda$  leads to higher ratios, aligning with expectations. However, since the training losses for  $\lambda=1$  and  $\lambda=2$  are comparably effective, we have chosen to implement  $\lambda=1$  in the training of our Skywork-MoE model.

### 4.2 Adaptive Auxiliary Loss Coefficients

The primary purpose of integrating an auxiliary loss (4) is to facilitate a balanced distribution of workload across experts during training. This balance not only ensures effective training for each expert but also fosters diversity among them. The intensity of this load balance regularization is governed by a tunable hyperparameter,  $\alpha$ , which is commonly set to either 1e-2 or 1e-3 in practical applications.

We present two key observations. Firstly, since each gating layer possesses its independent auxiliary loss, the coefficients corresponding to these losses do not necessarily have to be identical. In that regard, a more explicit form of the total loss (5) should be

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \sum_{l=1}^{M} \alpha^{(l)} \mathcal{L}_{\text{aux}}^{(l)},$$

where M is the total number of MoE layers, and  $\mathcal{L}_{\text{aux}}^{(l)}$  and  $\alpha^{(l)}$  are auxiliary loss and its coefficient for the l-th MoE layer, respectively. We speculate that there may exist a combination of "optimal" coefficient values that is superior to a single fixed global auxiliary loss coefficient applicable to all layers.

Secondly, if the load is already balanced across the experts during training, then it is advisable to reduce the auxiliary loss coefficients to alleviate the load balance regularization. On the contrary, in scenarios where there is a significant imbalance in load distribution among experts, increasing the coefficients would enforce stricter load balance regularization. The rationale for adjusting these coefficients is primarily to prioritize the optimization of the crossentropy loss for next-word prediction, while treating load balance regularization as a secondary, potentially counterproductive, goal.

To address this, we propose the method of *Adaptive Auxiliary Loss Coefficients*. This approach involves monitoring the token drop rate,

which we use as a measure for expert load balance, for each MoE layer throughout the training process, and adaptively updating the coefficients for subsequent iterations based on the observed token drop rates. The updates to the loss coefficients are designed to be positively correlated with the token drop rates.

More specifically, we define the update mechanism as follows:

$$\hat{\alpha}_{i+1}^{(l)} = f(d_i^{(l)}), \tag{7}$$

$$\alpha_{i+1}^{(l)} = \beta \alpha_i^{(l)} + (1-\beta)\hat{\alpha}_{i+1}^{(l)}, \qquad (8)$$

where:

- f is an increasing function mapping the current observed token drop rate  $d_i^{(l)}$  to an estimated auxiliary loss  $\hat{\alpha}_{i+1}^{(l)}$  for the next iteration
- $\alpha_{i+1}^{(l)}$  represents the moving average of  $\hat{\alpha}_{i+1}^{(l)}$ , serving as the actual auxiliary loss coefficient for the next iteration. This moving average approach mitigates abrupt changes in regularization intensity.
- $\beta$ , a parameter within the range (0, 1), balances the weight between the existing moving average and the new estimate.

In our specific implementation, we define  $f(d) = \xi d$  for some  $\xi > 0$ , with the constraint that f(d) does not exceed a maximum value  $c_{\text{max}}$ . This results in a piece-wise linear function:

$$f(d) = \begin{cases} \xi d & \text{if } d \le \alpha_{\text{max}}/\xi, \\ \alpha_{\text{max}} & \text{if } d > \alpha_{\text{max}}/\xi. \end{cases}$$
(9)

The hyper-parameter  $\xi$  regulates the sensitivity of the loss coefficients to the token drop rate. During our training of the Skywork MoE model, we set  $\xi=1/5$ ,  $\alpha_{\rm max}=0.01$ , and  $\beta=0.99$ . This configuration effectively maintained both token drop rates and auxiliary loss coefficients at desirable levels.

#### 5 Skywork-MoE

Skywork-MoE is a massive MoE model with a total of 146 billion parameters and 22 billion activated parameters. It initialized from our in-house pre-trained Skywork-13B (Wei et al.,

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Figure 3: Top left: Training loss curves for MoE models with and without gating normalization, illustrating that gating normalization contributes to a moderate improvement in loss. Top right: Evolution of the token drop rate for each model, showing the regularization effect of gating normalization which helps to reduce token drop during gating. Lower: Ratios Max1/Max<sup>2</sup> and Max2/Max<sup>3</sup> from the softmax output of the 3rd gating layer throughout training. We observe that a higher std parameter value increases both ratios as expected. For the model trained without gating normalization, both ratios converge to one (indicated by the horizontal dashed line), a condition considered detrimental for the model's performance.

[2023\)](#page-9-7) dense checkpoint[2](#page-6-1) , and is trained with gating logit normalization and adaptive auxiliary loss coefficient.

Skywork-MoE has undergone several stages of training, each characterized by a unique learning rate schedule and composition of training data. The data utilized to train Skywork-MoE consists of a curated subset of our SkyPile corpus [\(Wei et al.,](#page-9-7) [2023\)](#page-9-7), enriched with a significant volume of synthetic data. Overall, the collective distribution of training data aligns with a ratio of approximately 7:2:1 among English, Chinese, and code data.

To evaluate the performance of Skywork-MoE, we consider the following popular benchmarks: To assess the model's knowledge and problem-solving skills in Chinese, we utilized the CEVAL [\(Huang et al.,](#page-8-11) [2023\)](#page-8-11) and CMMLU

[\(Li et al.,](#page-9-10) [2023\)](#page-9-10) benchmarks. The MMLU [\(Hendrycks et al.,](#page-8-12) [2021a\)](#page-8-12) benchmark was chosen to evaluate English proficiency. For testing mathematical reasoning, the GSM8K [\(Cobbe](#page-8-13) [et al.,](#page-8-13) [2021\)](#page-8-13) and MATH [\(Hendrycks et al.,](#page-8-14) [2021b\)](#page-8-14) datasets were included. Additionally, the model's programming capabilities were assessed using the HumanEval [\(Chen et al.,](#page-8-15) [2021\)](#page-8-15) dataset.

We also present benchmark results for recent open-source models of comparable size, encompassing both dense and MoE architectures. Those models include: Deepseek-67B [\(DeepSeek-AI,](#page-8-16) [2024a\)](#page-8-16), Qwen1.5-72B [\(Qwen Team,](#page-9-11) [2023\)](#page-9-11), Llama2-70B [\(Touvron](#page-9-2) [et al.,](#page-9-2) [2023b\)](#page-9-2), Llama3-70B [\(Meta-AI,](#page-9-3) [2024\)](#page-9-3), Mixtral 8\*7B [\(Mistral-AI,](#page-9-12) [2023\)](#page-9-12), Mixtral 8\*22B [\(Mistral-AI,](#page-9-13) [2024\)](#page-9-13), DBRX-Instruct [\(Databricks,](#page-8-17) [2024\)](#page-8-17), Deepseek-V1 [\(Dai et al.,](#page-8-5) [2024\)](#page-8-5), Deepseek-V2 [\(DeepSeek-AI,](#page-8-2) [2024b\)](#page-8-2).

The evaluation results are presented in Table [1.](#page-7-0) It can be seen that Skywork-MoE

<span id="page-6-1"></span><sup>2</sup>The open sourced version of Skywork-13B has been trained for 3.2 trillion tokens. the in-house version has undergone additional pre-training on an extra 2 trillion tokens.

<span id="page-7-0"></span>

|               | # <b>AP</b> | #TP | CEVAL | CMMLU | MMLU | GSM8K | MATH | HumanEval |
|---------------|-------------|-----|-------|-------|------|-------|------|-----------|
| Deepseek-67B  | 67          | 67  | 66.1  | 70.8  | 71.3 | 63.4  | 18.7 | 42.7      |
| Qwen1.5-72B   | 72          | 72  | 84.1  | 83.5  | 77.5 | 79.5  | 34.1 | 41.5      |
| Llama2-70B    | 70          | 70  | -     | -     | 68.9 | 56.8  | 13.6 | 29.9      |
| Llama3-70B    | 70          | 70  | -     | -     | 78.8 | 82.7  | 36.7 | 39.0      |
| Mixtral 8*7B  | 13          | 47  | -     | -     | 70.6 | 58.4  | 28.4 | 40.2      |
| Mixtral 8*22B | 39          | 141 | -     | -     | 77.8 | 78.6  | 41.8 | 45.1      |
| Grok-1        | 86          | 314 | -     | -     | 73.0 | 62.9  | 23.9 | 63.2      |
| DBRX-Instruct | 36          | 132 | -     | -     | 73.7 | 66.9  | -    | 70.1      |
| Deepseek-V2   | 21          | 236 | 81.7  | 84.0  | 78.5 | 79.2  | 43.6 | 48.8      |
| Skywork-13B   | 13          | 13  | 62.1  | 62.4  | 62.7 | 60.2  | 8.4  | 18.9      |
| Skywork-MoE   | 22          | 146 | 82.2  | 79.5  | 77.4 | 76.1  | 31.9 | 43.9      |

Table 1: Evaluation results of Skywork-MoE on popular LLM benchmarks. Results of recent open models are also reported for comparison. The columns titled "#AP" and "#TP" stand for the number of activated parameters and that of total parameters (in billion), respectively.

![](_page_7_Figure_2.jpeg)

Figure 4: The curves of token drop rate (top) and those of auxiliary loss coefficient (bottom) for all gating layers during the pre-training of our Skywork-MoE. It can be seen that the auxiliary loss coefficients is responsive to the change in token drop rates.

achieves strong scores of 82.2 and 79.5 on the CEVAL and CMMLU benchmarks, respectively, surpassing Deepseek-67B, and is closely trailing behind Deepseek-V2. On the MMLU, Skywork-MoE scores 77.4, which is competitive when compared to higher-capacity models like Qwen1.5-72B and slightly lower than Llama3-70B. In mathematical related tasks (GSM8K and MATH), Skywork-MoE's scores of 76.1 and 31.9 are notable. It comfortably outperforms Llama2-70B and Mixtral 8\*7B and stands close to larger models such as Deepseek-V2 (79.2 and

43.6). This highlights the model's ability to handle complex quantitative and logical reasoning, a challenging area for many language models. On the HumanEval benchmark, which tests code synthesis capabilities, Skywork-MoE scores 43.9. This is a strong performance, exceeding all dense models in our comparison. It is slightly below Deepseek-V2, suggesting room for improvement in programming-related tasks. Overall, it is pertinent to conclude that our Skywork-MoE outperforms Deepseek-67B and Llama2-70B, but trails behind Llama3-70B and several larger MoEs such as Mixtral 8\*22B and Deepseek-V2.

#### 6 Conclusion

In this work we introduced the techniques and insights we gained behind the development of the Skywork-MoE model. Our comparative analysis of upcycling pre-existing models versus training from scratch provides insights and guidelines into the initization decisions required for MoE model development. This understanding allows for more informed and effective planning and allocation of resources in large-scale MoE training projects. We introduced gating logit normalization and adaptive auxiliary loss coefficients, two techniques that have notably enhanced expert diversification and provided a flexible framework for adjusting auxiliary losses, respectively. Based on these findings, we trained Skywork-MoE, an open-source MoE upcycled from previous Skywork-13B checkpoint. Its strong performance validates the effectiveness of our approach.

