# 4 METHOD

As discussed in Sec. [3,](#page-2-0) our analysis reveals three key observations. First, both the 32B and 7B models benefit from explicit perception prompting, with the 7B model showing greater potential for improvement. Second, the 7B model's baseline visual perception is relatively weak, requiring additional supervision to reliably ground visual information. Third, responses to textual perception prompts are unstable, indicating the need for better perception alignment.

Motivated by these insights, we propose a two-stage training pipeline, VTPerception-R1, to systematically enhance multimodal perception, particularly for the 7B variant. The overall framework is illustrated in Fig. [1.](#page-4-0)

*Stage 1*: Visual Perception–Augmented Supervised Fine-Tuning (SFT). This stage trains the model to generate focused ¡description¿ fields, improving its ability to extract and articulate visual content and grounding reasoning inputs.

*Stage 2*: Perception-Aware Reinforcement Learning (RL). Building on Stage 1, this stage refines reasoning with a composite reward that encourages (i) capturing key evidence in <description>and <think>steps, and (ii) logical consistency between evidence and answers.

This design follows two principles: (a) Make perception explicit before reasoning to improve grounding and transparency, and (b) Reward what matters for reasoning to ensure faithful, consistent outputs. Empirically, VTPerception-R1significantly boosts evidence recall, reasoning consistency, and answer quality, especially for perception-deficient smaller models.

#### 4.1 STAGE I: PERCEPTION-AUGMENTED SFT

**Data and template.** Starting from LLaVA-CoT style data, we convert each instance to a structured target:

$$<\!\!description\!\!>\!\ldots<\!\!/description\!\!>\!<\!\!think\!\!>\!\ldots<\!\!/think\!\!>\!<\!\!answer\!\!>\!\ldots<\!\!/answer\!\!>\!\ldots<\!\!/answer\!\!>\!\ldots$$

Here, <description>summarizes only the visual/textual evidence that is relevant to the question and useful for reasoning (not a generic caption). We retain the original chain-of-thought inside <think>and the final solution in <answer>. This explicit structuring encourages the model to first ground the input in perception, then conduct reasoning in a disentangled manner.

**Objective.** We minimize token-level cross-entropy over the full target sequence:

$$\mathcal{L}_{SFT} = -\sum_{t} \log \pi_{\theta} (y_t \mid x, y_{< t}), \tag{2}$$

which bootstraps the model to (i) attend to question-relevant regions, (ii) verbalize them succinctly in <description>, and (iii) reason over these cues. This establishes a stable perception-to-description interface that Stage II can further refine.

**Motivation.** Directly training on raw CoT-style data often entangles perception, reasoning, and answering, making it difficult for the model to learn robust visual grounding. By inserting the <description>stage, we encourage the model to first distill multimodal evidence into a compact, interpretable representation. This intermediate step reduces spurious correlations, improves generalization, and provides a controllable interface that can be explicitly inspected or modified during inference. In practice, the <description>field resembles a structured "scratchpad" for perception, while the subsequent <think>step focuses purely on abstract reasoning.

**Effect.** This stage produces a model that is able to (i) highlight salient objects and attributes, (ii) capture their spatial or semantic relations, and (iii) chain them into reasoning sequences leading to the final answer. Empirically, we find that Perception-Augmented SFT improves consistency between visual grounding and reasoning, and also simplifies downstream alignment, since the model is already trained to organize its outputs into modular components.

#### 4.2 STAGE II: PERCEPTION-AWARE RL

**Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)**. DAPO is a reinforcement learning algorithm designed for large-scale reasoning models with long chain-of-thought outputs. Compared to earlier methods such as GRPO, it improves both stability and efficiency by introducing four key techniques: asymmetric clipping, dynamic sampling, token-level optimization, and overlong reward shaping. Formally, for each prompt x, G responses  $\{o_i\}_{i=1}^G$  are sampled, and group-relative advantages are computed to normalize rewards. The training objective is a clipped-ratio policy gradient:

$$J(\theta) \ = \ \mathbb{E}_{\{o_i\} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{\sum_i |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min \left( r_{i,t}(\theta) \, \widehat{A}_{i,t}, \text{ clip} \left( r_{i,t}(\theta), \, 1 - \varepsilon_{\text{low}}, \, 1 + \varepsilon_{\text{high}} \right) \, \widehat{A}_{i,t} \right) \right],$$

where  $r_{i,t}(\theta)$  is the token-level importance ratio and  $\widehat{A}_{i,t}$  the normalized advantage. To avoid degenerate updates, DAPO excludes trivial groups where all rollouts are correct or all are wrong.

These improvements allow DAPO to provide stronger learning signals for long reasoning traces, improving convergence and reducing instability. In our work, DAPO serves as the backbone optimization method, enabling perception-aware rewards to directly shape both description quality and reasoning accuracy.

**Perception-Aware Reward**. As is described in Section A.1, we motivated the model to enhance the capability of description before think answer. We want the model to reason according to the information they have seen. What is more, the combined attention of textual information and visual information are all important. The total reward is a weighted sum:

$$R = R_{\text{acc}} + R_{\text{fmt}} + R_{\text{vkey}} + R_{\text{tkey}} + R_{\text{rep}} + R_{\text{cons}}. \tag{3}$$

We describe each component below.

- (1) Answer accuracy (Racc). This reward measures whether <answer> matches the ground truth, providing a direct, task-specific signal that reinforces correct reasoning and supports perceptionrelated rewards.
- (2) Format compliance (Rfmt). This reward enforces the template <description>→ <think>→ <answer>, penalizing missing or repeated segments to ensure consistent, perception-grounded reasoning.
- (3) Repetition control (Rrep). This component penalizes repeated n-grams to discourage verbosity and promote concise, evidence-rich <description> and <think> segments.
- (4) Visual key-info (Rvkey). We enhance the data with key visual cues that are directly relevant to the task, ensuring the model captures meaningful visual information. The reward measures how well <description> covers these critical visual elements. Given a key set K and the set of facts D extracted from <description>, we compute the recall cov = |K∩D| |K| and discretize it into three levels:

$$R_{\text{vkey}} = \begin{cases} 1.0, & \text{if } \text{cov} \ge \tau_{\text{hi}}, \\ 0.5, & \text{if } \tau_{\text{lo}} \le \text{cov} < \tau_{\text{hi}}, \\ 0.0, & \text{otherwise.} \end{cases}$$
(4)

This reward encourages the model to capture task-relevant visual evidence faithfully, thereby improving its ability to perceive and utilize visual information in downstream reasoning.

(5) Textual key-info (Rtkey). To strengthen textual perception, we augment training data with taskspecific key information—such as OCR text, numerical values, unit constraints, symbols, and relevant commonsense cues. The reward measures how well <think> incorporates these critical textual elements. Given a key set K and the set of facts D extracted from <think>, we compute the recall cov = |K∩D| |K| and discretize it into three levels:

$$R_{\text{tkey}} = \begin{cases} 1.0, & \text{if } \text{cov} \ge \tau_{\text{hi}}, \\ 0.5, & \text{if } \tau_{\text{lo}} \le \text{cov} < \tau_{\text{hi}}, \\ 0.0, & \text{otherwise.} \end{cases}$$
 (5)

As a complementary signal to Rvkey, this reward ensures that reasoning does not ignore textual evidence and helps maintain a balance between visual and textual perception.

(6) Desc–reasoning consistency (Rcons). Finally, we add a consistency reward to align perception with reasoning. It checks whether the entities, attributes, and numerical values referenced in <think> and <answer> are supported by the evidence stated in <description> and the question. Let Fans denote the set of key items extracted from <think> + <answer>, and E the items from <description> + question, defined as

$$cons = \frac{|F_{ans} \cap E|}{\max(1, |F_{ans}|)}, \qquad R_{cons} = \begin{cases} 0, & \text{if clear conflicts exist,} \\ cons, & \text{otherwise.} \end{cases}$$
 (6)

Clear conflicts (e.g., mismatched numbers or attributes) score zero. Rcons rewards alignment between reasoning and perceived evidence, reducing hallucinations and ensuring grounded answers.

