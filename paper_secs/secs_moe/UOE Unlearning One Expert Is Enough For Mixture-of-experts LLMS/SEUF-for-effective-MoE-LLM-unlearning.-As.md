# **SEUF for effective MoE LLM unlearning.** As

discussed earlier, a new paradigm tailored for MoE LLM unlearning is urgently needed to address the challenges of unintentional expert selection shifts in routers and excessive unlearning of non-target experts. Therefore, we propose a framework that (1) identifies the most relevant target experts, (2) ensures that these target experts remain highly activated throughout the unlearning process to avoid selection shifts, and (3) limits the impact of unlearning on non-target experts. Spurred by these, we introduce SEUF, where unlearning is confined to M most relevant target experts. We refer the readers to Alg. 1 for an illustration of SEUF.

This approach starts with an expert attribution process to accurately identify the most M relevant experts for the unlearning task (step 1-3). Then, the gradient computation selected experts  $e_M$  and their corresponding routers  $R_{e_M}$  are enabled (step 4), while other parameters are frozen. Step 5 performs unlearning using any unlearning approach, as our framework is flexible. For example, gradient ascent can be applied with our defined loss functions. Next, we present the details of the expert attribution process and define the anchor loss function.

#### <span id="page-5-0"></span>Algorithm 1 SEUF Unlearning Algorithm

**Output:** Unlearned model  $\theta_u$ 

**Input:** Pretrained model  $\theta_o$ , forget set  $\mathcal{D}_f$ , retain set  $\mathcal{D}_r$ , **Setup:** Retain loss  $\ell_r$ , forget loss  $\ell_f$ , anchor loss  $L_{\rm anchor}$ , the number of experts to select M

1:  $\mathcal{D}_s \leftarrow \text{Sample\_Subset}(\mathcal{D}_f)$ 

2:  $s \leftarrow \text{Record\_Affinity\_Score}(\boldsymbol{\theta}_o, D_s)$ 

3:  $e_M \leftarrow \text{Ranking\_And\_Select}(s, M)$ 

4: Activate\_Expert\_And\_Router( $\theta_o, e_M, R_{e_M}$ )

5:  $\theta_u \leftarrow \text{Unlearn}(\theta_o, \ell_f(\mathcal{D}_f), \ell_r(\mathcal{D}_r), L_{\text{anchor}})$ 

6: Return  $\theta_u$ 

**♦ Expert attribution.** While the token assignment ratio for each expert (shown in Fig. 2), can serve as a basic attribution metric, it overlooks finer details that are important for precise comparisons, due to the hidden states in each layer summed by weighted average. To address this, we adopt a gating score-based task affinity calculation method from (Wang et al., 2024b). Specifically, the affinity score for the i-th expert  $e_i^{(l)}$  in the l-th layer of an MoE LLM is defined as:

<span id="page-5-1"></span>
$$s_i^{(l)} = \frac{1}{Z} \sum_{j=1}^{Z} \frac{1}{L_j} \sum_{t=1}^{L_j} g_{i,t}^{(l)}$$
 (2)

where Z is size of the calibration dataset used for expert attribution,  $L_j$  represents the length of the j-th input sequence  $\mathbf{x}_j$ , and  $g_{i,t}^{(l)}$  is the probability score assigned to expert  $\mathbf{e}_i^{(l)}$  for the t-th token. Following Wang et al. (2024b), the attribution data can be a subset universally sampled from the original forget set. We find that a subset containing over 100,000 tokens is robust enough to select the most relevant experts for an unlearning task. For each layer, we rank the experts based on their affinity score and then finally select the top M experts as the target expert for unlearning ( $e_M$  in Algo. 1).

◆ Router anchor loss. A key challenge in unlearning is the expert selection shift, where the true target experts are hidden by the routers, while less relevant experts are activated during inference and inadvertently involved in the unlearning process. To mitigate this, we propose the router anchor loss, which encourages the previously identified target expert to remain consistently activated throughout unlearning. The loss is formulated as:

$$L_{\text{anchor}}^{(l)} = \|\mathbf{g}^{(l)} - [a_1^{(l)}, a_2^{(l)}, \dots, a_{E^{(l)}}^{(l)}]\|_2^2, \tag{3}$$

where  $E^{(l)}$  is the total number of experts in the l-th layer,  $\mathbf{g}^{(l)} = [g_1^{(l)}, g_2^{(l)}, \dots, g_i^{(l)}]$  is the output of router, and  $a_i^{(l)} = 1$  if the i-th expert is identified as the target expert, otherwise  $a_i^{(l)} = 0$ . The unlearning loss can then be formularized as:

<span id="page-5-3"></span>
$$\min_{\boldsymbol{\theta}} \ell_f(\boldsymbol{\theta}; \mathcal{D}_f) + \lambda \ell_r(\boldsymbol{\theta}; \mathcal{D}_r) + \alpha L_{\text{anchor}}^{(l)}, \tag{4}$$

<span id="page-5-2"></span>Table 2: Model utility (UT $\uparrow$ ) comparison at the same level of forget efficacy (FE $\approx$  0.25), when the top M experts from either the same layer or different layers in DeepSeek are unlearned using GA on WMDP benchmark, also when 4 shared experts are included.

| Selected experts | Top-1  | Top-3  | Top-6  | Top-1+4-shared |
|------------------|--------|--------|--------|----------------|
| Same layer       | 0.5100 |        | 0.4652 | 0.3554         |
| Different layers | 0.5100 | 0.2852 | 0.2567 | _              |

where  $\alpha$  controls the strength of anchor loss. Its sensitivity is analyzed in Appendix Sec. B.

 $\bullet$  Selection of top M experts. When forming  $e_M$  of the top M experts, there are two approaches: 1) selecting the top M experts from all experts across all layers based on the affinity score  $s_i^{(l)}$ in Eq.2; and 2) to mitigate selection shift from previous layers, another approach is to choose the top M experts from the same layer. We examined both approaches under different settings M=1,3,6, and present the results in Tab. 2. We observe that unlearning a single expert (M=1) yields better performance than unlearning multiple experts, regardless of whether they come from the same layer or different layers. This trend of single-expert unlearning yielding the best performance is also observed across other unlearning tasks (see Tab. 7 in Appendix). This suggests:

**Insight 4**: Unlearning top-1 expert is the most effective.

From Tab. 2, we also observe that unlearning multiple experts across different layers leads to a substantial performance decline. To further analyze the Insight 4, let the total gradient update during unlearning be:  $\Delta W = \sum_{i \in e_M} \lambda_i \nabla \mathcal{L}_i$ , where  $e_M$ is the set of selected experts being unlearned,  $\lambda_i$ denotes their contribution weight, and  $\nabla \mathcal{L}_i$  is their corresponding gradient update in Eq. (4). When only the top-1 expert is selected for unlearning, the modification to the weights remains minimal, ensuring low gradient interference. For multiple experts within the same layer, the gradient updates may partially cancel out, leading to moderate disruption. However, for multiple experts across different layers, the gradient updates affect distinct feature hierarchies, resulting in an unstable gradient flow and widespread model disruption.

This analysis also explains the deficiency of unlearning shared experts. In a given layer, shared experts are activated for all tokens, making them intuitively suitable targets for unlearning. However, Tab. 2 shows that unlearning the top-1 expert along with 4 shared experts causes a greater utility drop than unlearning top-6 experts in the same layer. Shared experts influence a broader range

of token representations, so making them active for unlearning triggers high-magnitude gradient updates across multiple pathways. Also, since shared experts consolidate common knowledge across diverse contexts [\(Liu et al.,](#page-9-4) [2024a\)](#page-9-4), their modification disrupts the model more severely, making them suboptimal for unlearning.

