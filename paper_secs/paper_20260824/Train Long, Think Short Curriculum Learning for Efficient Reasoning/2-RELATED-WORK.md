# 2 RELATED WORK

Test-Time Scaling and the Rise of Long-Chain Reasoning. A dominant trend in enhancing the reasoning capabilities of LLMs is increasing computation at inference time. This strategy, often termed test-time scaling, has consistently improved performance in complex reasoning tasks, from mathematics to code generation [\(Wang et al.,](#page-13-0) [2023;](#page-13-0) [Wu et al.,](#page-13-1) [2025a;](#page-13-1) [Wei et al.,](#page-13-2) [2022\)](#page-13-2). Prominent approaches include sampling multiple reasoning paths and selecting the most consistent answer (self-consistency) [\(Wang et al.,](#page-13-0) [2023\)](#page-13-0), exploring solution paths with tree-based search [\(Yao et al.,](#page-13-3) [2023\)](#page-13-3), and iterative refinement [\(Madaan et al.,](#page-12-0) [2023;](#page-12-0) [Welleck et al.,](#page-13-4) [2024\)](#page-13-4). Recent state-of-the-art reasoning models, such as OpenAI's O1 and DeepSeek's R1-style models, are trained with reinforcement learning to generate extended reasoning traces, embodying a "think more" philosophy to tackle difficult problems [\(Jaech et al.,](#page-12-1) [2024;](#page-12-1) [Guo et al.,](#page-12-2) [2025\)](#page-12-2). However, this paradigm often leads to significant computational overhead and a phenomenon known as "overthinking," where models produce verbose and inefficient reasoning chains even for simple problems [\(Anonymous,](#page-12-3) [2025\)](#page-12-3). These methods, while powerful, typically lack precise mechanisms to control the length of their outputs, creating a trade-off where higher accuracy comes at the cost of unpredictable and often excessive token usage.

Approaches to Length Control and Reasoning Efficiency. In response to the inefficiency of long-chain reasoning, a parallel line of research has focused on controlling the length of LLM outputs. Early work in this area addressed general text generation through architectural modifications [\(Butcher et al.,](#page-12-4) [2025\)](#page-12-4) or fine-tuning on instruction datasets labeled with desired lengths [\(Yuan et al.,](#page-13-5) [2024\)](#page-13-5). More recent work has tailored length control specifically for reasoning. Some approaches train models to generate shorter chains of thought (Arora & Zanette, 2025; Kang et al., 2025), while others use "budget-forcing" techniques that truncate outputs or pad with special tokens to meet a fixed limit (Muennighoff et al., 2025). However, these hard constraints can be suboptimal, as abrupt truncation can disrupt reasoning.

Other methods pursue finer-grained control by identifying and suppressing low-utility tokens at inference time. Xia et al. (2025) propose *TokenSkip*, a method that estimates the importance of the token and skips useless tokens to compress reasoning chains while preserving performance. Xu et al. (2025) present the *Chain of Draft* strategy, prompting models to write concise intermediate drafts rather than verbose step-by-step thoughts, dramatically reducing token usage without sacrificing accuracy. Wu et al. (2025b) provide a complementary analytical perspective, showing that accuracy follows an inverted-U curve with respect to chain length and proposing a length-aware voting heuristic that filters out traces that are too short or too long.

Reinforcement learning has also been applied to dynamically optimize reasoning length. Fang et al. (2025) introduce *Thinkless*, a policy learning framework that trains models to decide *when to think*, selecting between short and long reasoning paths based on the difficulty of the problem. Similarly, Dumitru et al. (2025) propose *ConciseRL*, which rewards models for generating correct but concise reasoning by incorporating a learned conciseness score into the RL reward function. These methods demonstrate that length control can be learned in a context-sensitive and adaptive way, enabling models to use fewer tokens on easier problems and longer chains only when necessary.

A more sophisticated approach is taken by Aggarwal & Welleck (2025) with Length Controlled Policy Optimization (LCPO). Their method uses reinforcement learning to train a model to adhere to a user-specified length budget provided directly in the prompt. The reward function optimizes for both task correctness and adherence to the target length, producing a model, L1, that can flexibly trade off accuracy and computational cost at inference time. This allows a user to request reasoning of a specific length (e.g., 512, 1024, or 2048 tokens) and receive a response that respects that budget. These methods, while offering powerful inference-time flexibility, treat length as a user-controlled parameter for a pre-trained, versatile model.

**Positioning Our Work.** Our work, "Train Long, Think Short," introduces a novel perspective by framing efficient reasoning as a *curriculum learning* problem. In contrast to previous methods such as LCPO (Aggarwal & Welleck, 2025), which train a model to respond to *user-specified length budgets at inference time*, our work investigates the training dynamic itself as a mechanism for optimization. We propose a dynamic training strategy where the budget is not a user-controlled parameter. Instead, it starts with a generous token budget that lets the model freely **explore** long reasoning paths, and then monotonically **decays** this budget so the same policy learns to *compress* its successful strategies into a concise form. The result is a standalone model that targets a tight token budget and, in practice, stays within roughly 5% of that limit on average, achieving substantial cost savings without any runtime user hints or prompt overhead.

### <span id="page-2-0"></span>3 METHODOLOGY

We build on Group Relative Policy Optimization (GRPO) and introduce a curriculum for length-controlled reasoning, augmented with explicit formatting and correctness signals. Our training signal for each generated completion combines three components: (1) a *correctness* reward based on automated verification, (2) a *length* reward encouraging adherence to a (curriculum-decayed) token budget, and (3) a *formatting* reward enforcing structured reasoning and answer separation via special tags. We first review the math behind GRPO, then formalize the prompt, define each reward component with its weighting, describe the curriculum schedule, and finally give the full optimization objective with refinements.

### 3.1 GRPO PRELIMINARIES

Given a prompt s, the current (old) policy  $\pi_{\theta_{\text{old}}}$  is used to sample a group of G responses  $\{a_i\}_{i=1}^G$ . Each response  $a_i$  is assigned a scalar reward  $r_i$  (defined below). Let the empirical mean and standard

#### <think>First, let's determine the total number of pizza slices available. Since there are 7 pizzas and each pizza is cut into 8 slices, we can calculate the total number of slices by multiplying the number of pizzas by the number of slices per pizza: Total slices = Number of pizzas × Slices per pizza Total slices = 7 × 8 Total slices = 56 Now that we know there are 56 slices in total, we need to find out how many slices each person can have if they want to share the pizzas equally. There are 4 people in total (Henry and his 3 friends), so we can divide the total number of slices by the number of people: Slices per person = Total slices ÷ Number of people Slices per person = 56 ÷ 4 Slices per person = 14 So, each person can have 14 slices of pizza if they want to share the pizzas equally.</think> **<answer>14</answer> Length Reward Function Budget Curriculum** friends want to share the pizzas equally, how many slices can each of them have? <think>Let's calculate: 7 pizzas \* 8 slices = 56 slices. 56 slices / 4 people = 14 slices per person.</think> **<answer>14</answer>**

**Question:** Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his

Figure 1: Curriculum Learning GRPO Overview. Our proposed setting performs GRPO with a length reward being applied to the generated thinking trace. The budget is decayed exponentially with a user specific decay factor and decay interval. In this example the decay factor γ is set to 0.7 and the decay interval T is set to 100. An initial budget of 256 tokens is given at start and decayed later down to 30. The figure demonstrates that the model learns to answer the same question with a way smaller token budget reaching the same solution.

deviation over the group be

$$\mu = \frac{1}{G} \sum_{i=1}^{G} r_i, \qquad \sigma = \sqrt{\frac{1}{G} \sum_{i=1}^{G} (r_i - \mu)^2 + \epsilon_{\text{stab}}},$$

where ϵstab > 0 is a small stabilizer to avoid division by zero. The group-relative advantage is

$$A_i = \frac{r_i - \mu}{\sigma}.$$

Define the probability ratio

$$r_i^{\text{ratio}} = \frac{\pi_{\theta}(a_i \mid s)}{\pi_{\theta_{\text{old}}}(a_i \mid s)}.$$

The clipped surrogate GRPO objective with reference regularization is as follows:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_s \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( r_i^{\text{ratio}} A_i, \text{ clip}(r_i^{\text{ratio}}, 1 - \epsilon, 1 + \epsilon) A_i \right) \right] - \beta \operatorname{KL}(\pi_{\theta} \parallel \pi_{\text{ref}}),$$

where ϵ > 0 controls the clipping window and β trades off deviation from a stable reference policy πref.

### 3.2 PROMPT STRUCTURE

To explicitly separate internal reasoning from the final answer and to enforce a fixed-length constraint, we prompt the model with the following instruction:

### **Prompt Template**

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process here
<answer>answer
<answer>answer
here</answer>answer
<answer>answer
here</answer>answer
<answer>answer</answer</answer>answer</a>
Assistant:

The ideal model output takes the form:

 $\langle <$ think> $\rangle$  chain-of-thought reasoning  $\langle <$ /think> $\rangle$   $\langle <$ answer> $\rangle$  final answer  $\langle <$ /answer> $\rangle$ , and the total token count is guided toward the budget via the curriculum and associated reward.

#### 3.3 REWARD DECOMPOSITION AND WEIGHTING

For each sampled response  $a_i$  of length  $\ell_i$  (in tokens), we define the total scalar reward as a weighted sum of three components:

$$r_i = \lambda_c \cdot r_i^{\text{correct}} + \lambda_\ell \cdot R_{\text{len}}(\ell_i) + \lambda_f \cdot R_{\text{fmt}}(a_i),$$

where  $\lambda_c$ ,  $\lambda_\ell$ ,  $\lambda_f$  are nonnegative scalar weights controlling the relative importance of correctness, length adherence, and formatting, respectively. This makes explicit the experimental 'weights' (e.g., correctness vs. length vs. formatting) used in different settings.

**Correctness Reward.** Let  $c_i \in \{0,1\}$  be the indicator that the final answer (extracted from within  $\langle answer> \rangle$ ) passes the automated verifier (math-verify)—either exact numeric/symbolic match or a graded acceptance if extended. Then:

$$r_i^{\text{correct}} = R_{\text{cor}} \cdot c_i,$$

where  $R_{\rm cor}>0$  is the base correctness reward. Optionally, if the verifier provides partial scores or confidence,  $c_i$  can be softened to [0,1] and  $r_i^{\rm correct}$  adjusted accordingly.

**Length Reward.** Let the current target length be L=B(t) (see next subsection). We define a triangular (piecewise linear) reward that encourages matching L without encouraging trivial short or excessively long outputs:

$$R_{\mathrm{len}}(\ell) = \begin{cases} R_{\mathrm{max}} \cdot \frac{\ell}{L} & \text{if } 0 \leq \ell \leq L, \\ R_{\mathrm{max}} \cdot \left(1 - \frac{\ell - L}{L}\right) & \text{if } L < \ell \leq 2L, \\ 0 & \text{if } \ell > 2L, \end{cases}$$

where  $R_{\rm max}>0$  is the maximum length reward at  $\ell=L$ . This shape (ramp-up, plateau at peak, ramp-down, hard cutoff) encourages the model to use the budget efficiently. In practice, we clip  $\ell$  when computing the length if a generation exceeds 2L to avoid inflated computation; those responses receive zero for the length component.

**Formatting Reward.** Define indicators  $\mathbb{I}_{think}$  and  $\mathbb{I}_{answer}$  that equal 1 if the output contains well-formed, non-overlapping  $\langle think \rangle / \langle think \rangle$  and  $\langle answer \rangle / \langle answer \rangle$  spans, respectively, and zero otherwise. Then:

$$R_{\text{fmt}}(a) = \alpha_{\text{think}} \cdot \mathbb{I}_{\text{think}} + \alpha_{\text{answer}} \cdot \mathbb{I}_{\text{answer}},$$

with  $\alpha_{\text{think}}$ ,  $\alpha_{\text{answer}} > 0$  rewarding proper structural separation. This encourages the model to clearly expose its reasoning and final answer in the prescribed format.

### 3.4 CURRICULUM TOKEN BUDGET

We impose a curriculum on the allowable token budget so that it decays exponentially over training steps, enabling a natural transition from exploration (long, rich reasoning) to compression (concise reasoning under tight constraints). Starting from an initial budget B0, the budget at training step t is:

$$B(t) = \max\left(1, B_0 \cdot \gamma^{\left\lfloor \frac{t}{T} \right\rfloor}\right),$$

where γ ∈ (0, 1) is the decay factor and T is the interval (in steps) between budget updates. The target length L in Rlen is set to B(t), making the length reward progressively stricter as training progresses.

