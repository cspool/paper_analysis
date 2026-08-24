# 3 Method

We propose Adaptive Reasoning Model (ARM), a reasoning model designed to optimize effectiveness and efficiency by adaptively selecting reasoning formats. Specifically, ARM is trained in two stages: *1)* Stage 1: Supervised Fine-tuning (SFT) for Reasoning Formats Understanding: In this stage, we use 10.8K diverse questions, each annotated with solutions in four distinct reasoning formats, to fine-tune the model and build a foundational understanding of different reasoning strategies. *2)* Stage 2: Reinforcement Learning (RL) for Encouraging Efficient Format Selection: We adopt an adapted version of the GRPO algorithm, named Ada-GRPO, to train the model to be capable of selecting more efficient reasoning formats over solely *Long CoT*, while maintaining accuracy.

#### 3.1 Stage 1: SFT for Reasoning Formats Understanding

In this stage, we leverage SFT as a cold start to introduce the model to various reasoning formats it can utilize to solve problems.[3](#page-2-0) These formats include three efficient reasoning formats *Direct*

<span id="page-2-0"></span><sup>3</sup> In preliminary experiments, models without SFT failed to distinguish between the four reasoning formats, like producing mixed outputs, wrapping a *Short CoT* response using the special tokens intended for *Long CoT*.

Answer, Short CoT, and Code, as well as the elaborate reasoning format Long CoT. We use special tokens (e.g., <Code></Code>) to embrace thinking rationale. Specifically, 1) Direct Answer: This format provides a direct answer without any reasoning chain, making it the most efficient in terms of token usage. 2) Short CoT: This format begins with a short reasoning and then provides an answer, which has been proved effective in mathematical problems [49]. 3) Code: This format adopts code-based reasoning, which has proven effective across a variety of tasks due to its structured process [50; 51; 24]. 4) Long CoT: This format involves a more detailed, iterative reasoning process, thus incurs higher token usage. It is suited for tasks requiring advanced reasoning capabilities, such as self-reflection and alternative generation, where those more efficient formats fall short [31; 11; 56].

#### <span id="page-3-0"></span>3.2 Stage 2: RL for Encouraging Efficient Format Selection

After SFT, the model learns to respond using various reasoning formats but lacks the ability to adaptively switch between them based on the task (see Section 4.3 for details). To address this, we propose **Ada**ptive GRPO (**Ada-GRPO**), which enables the model to dynamically select appropriate reasoning formats according to the task difficulty through a format diversity reward mechanism.

**GRPO** In traditional GRPO [38], the model samples a group of outputs  $O = \{o_1, o_2, \cdots, o_G\}$  for each question q, where G denotes the group size. For each  $o_i$ , a binary reward  $r_i$  is computed using a rule-based reward function that checks whether the prediction pred matches the ground truth gt:

$$r_i = \mathbb{1}_{passed(gt, pred)}. \tag{1}$$

However, since traditional GRPO solely optimizes for accuracy, it leads, in our setting, to overuse of the highest-accuracy format while discouraging exploration of alternative reasoning formats. Specifically, if *Long CoT* achieves higher accuracy than other formats, models trained with GRPO tend to increasingly reinforce it, leading to an over-reliance on *Long CoT* and reduced exploration of more efficient alternatives. We refer to this phenomenon as **Format Collapse**, which ultimately hinders the model's ability to develop adaptiveness. We further analyze this in Section 4.3.

**Ada-GRPO** We propose Ada-GRPO to address the format collapse issue. Specifically, Ada-GRPO amplifies the reward  $r_i$  for less frequently sampled reasoning formats, preventing their disappearance and ensuring adequate learning. Formally, we scale the reward  $r_i$  to  $r_i'$  by:

$$r_i' = \alpha_i(t) \cdot r_i, \tag{2}$$

$$\alpha_i(t) = \frac{G}{F(o_i)} \cdot decay_i(t), \tag{3}$$

$$decay_i(t) = \frac{F(o_i)}{G} + 0.5 \cdot \left(1 - \frac{F(o_i)}{G}\right) \cdot \left(1 + \cos\left(\pi \cdot \frac{t}{T}\right)\right),\tag{4}$$

where  $F(o_i)$  denotes the number of times the reasoning format corresponding to  $o_i$  appears within its group O, and t represents the training step.  $\alpha_i(t)$  is a format diversity scaling factor that gradually decreases from  $\frac{G}{F(o_i)}$  at the beginning of training (t=0) to 1 at the end of training (t=T).

We introduce  $\alpha_i(t)$  to extend GRPO into **Ada-GRPO**, enabling models to adaptively select reasoning formats. Specifically,  $\alpha_i(t)$  consists of two components: 1) **Format Diversity Scaling Factor**  $\frac{G}{F(o_i)}$ : To prevent premature convergence on the highest-accuracy format (i.e., format collapse to  $Long\ CoT$ ), we upweight rewards for less frequent formats to encourage exploration. 2) **Decay Factor**  $decay_i(t)$ : To avoid long-term misalignment caused by over-rewarding rare formats, this term gradually reduces the influence of diversity over time. For example,  $\frac{G}{F(o_i)}$  might make the model favor a lower-accuracy format like  $Short\ CoT$  over  $Long\ CoT$  simply because it appears less frequently and thus receives a higher reward. While such exploration is beneficial early in training, it can hinder convergence later. The decay mechanism mitigates this by promoting diversity initially, then shifting focus to accuracy again as training progresses. Refer to Appendix A for details of the decay factor.

Then the group advantage  $\hat{A}_{i,k}$  for all tokens in each output is computed based on the group of reshaped rewards  $\mathbf{r}' = \{r'_1, r'_2, \cdots, r'_G\}$ :

<span id="page-3-1"></span>
$$\hat{A}_{i,k} = \frac{r'_i - \text{mean}(\{r'_1, r'_2, \cdots, r'_G\})}{\text{std}(\{r'_1, r'_2, \cdots, r'_G\})}.$$
(5)

Finally, we optimize the model by maximizing the following objective (see Appendix [A](#page-13-0) for details):

$$\mathcal{J}_{\text{Ada-GRPO}}(\theta) = \mathbb{E}\left[q \sim P(Q), \{o_{i}\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(O|q)\right] \left[\frac{1}{\sum_{i=1}^{G} |o_{i}|} \sum_{i=1}^{G} \sum_{k=1}^{|o_{i}|} \left\{\min\left[\frac{\pi_{\theta}(o_{i,k}|q, o_{i, < k})}{\pi_{\theta_{\text{old}}}(o_{i,k}|q, o_{i, < k})} \hat{A}_{i,k}, \right] - \beta \operatorname{KL}\left[\pi_{\theta} \parallel \pi_{\text{ref}}\right] \right\}\right].$$
(6)

