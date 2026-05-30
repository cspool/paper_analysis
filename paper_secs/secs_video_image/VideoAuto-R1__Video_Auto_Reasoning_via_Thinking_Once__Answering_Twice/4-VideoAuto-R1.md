# 4 VideoAuto-R1

In this section, we present **VideoAuto-R1**, a simple yet effective framework that reasons only when necessary, as illustrated in Figure 2. During training, we adopt an  $answer \rightarrow think \rightarrow answer$  template. At inference time, an early-exit mechanism determines whether to continue reasoning after the first answer.

### 4.1 Thinking Once, Answering Twice

A common approach to auto-thinking involves learning a mode-switching policy during training, e.g., randomly dropping CoT traces in SFT so the model alternates between direct and CoT outputs (Zhang et al., 2025b). While effective on text, it depends on careful data balancing and is sensitive to training hyperparameters. In video, the scarcity of high-quality reasoning examples further exacerbates instability.

We adopt a different perspective: genuine CoT should be built on top of an initial answer. For easy questions, the initial answer should suffice; for harder ones, the model should verify and revise its response within the same generation. Accordingly, we do not train separate "think" and "direct" modes. Instead, the model always learns to generate a concise first answer and a reasoned second answer. This design avoids the need for per-sample mode labels, specialized switch tokens or heads, or other artifacts. The distinction between direct and thinking modes is made solely at test time through a confidence-based early-exit mechanism.

**Output Format.** Given a prompt q, each training response o follows a strict, verifiable format:

$$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \$$

Here,  $a_1$  and  $a_2$  are short, verifiable answers, and r is a free-form rationale. We enforce exactly two \boxed{...} blocks and one <think>...</think> block, with no extra text before/after. To achieve such an output format, a system prompt (Table 2) is carefully designed, enabling generation without cold-start SFT.

**Fallback Tolerance.** For mathematically or symbolically complex problems, the model may be unable to produce a correct  $a_1$  without intermediate reasoning (Yue et al., 2025a). To prevent low-confidence guesses, we provide a designated fallback string. When immediate answering is infeasible, the model outputs "Let's analyze the problem step by step" in the first box, then proceeds to reasoning and produces the final answer  $a_2$ . This design preserves the output grammar, avoids spurious guesses, and ensures the early-exit mechanism remains unambiguous and interpretable.

Why "answer-think-answer"? This template decouples when to think, handled at test time by our early-exit rule, from how to think, namely the reasoning behavior learned during RL training. Empirically, this design yields more stable training for videos with less data effort than traditional mode-switching approaches (Zhang et al., 2025b). It also makes inference easy to control: with ample compute, one can always use the reviewed answer, while under tight budgets the model can fall back to the initial direct answer and still benefit from RL training. Overall, this decoupling of the training objective and inference policy gives users flexible control over the trade-off between accuracy and efficiency.

<span id="page-6-0"></span>**Table 2 System Prompt for VideoAuto-R1.** The prompt follows an  $answer \rightarrow think \rightarrow answer$  template, enabling both direct and CoT outputs in one generation.

#### SYSTEM PROMPT

You are a helpful assistant.

FIRST: Output your initial answer inside the first \\boxed{...} without any analysis or explanations. If you cannot determine the answer without reasoning, output \\boxed{Let's analyze the problem step by step.} instead.

THEN: Think through the reasoning as an internal monologue enclosed within <think>...</think>...

AT LAST: Output the final answer again inside \\boxed{...}. If you believe the previous answer was correct, repeat it; otherwise, correct it.

Output format: \\boxed{...}<\think>...</think>\\boxed{...}

### 4.2 Training: Dual-Answer Reward with GRPO

We follow the GRPO framework described in Section 3.1, but introduce a new *dual-answer* reward that supervises both the initial and reviewed answers. Let  $a_1$  and  $a_2$  denote the first and second boxed answers, respectively. The total reward is given by:

$$R = w_1 R_{\text{task}}^{(1)}(a_1) + w_2 R_{\text{task}}^{(2)}(a_2) + \lambda R_{\text{fmt}} + \alpha R_{\text{fallback}}$$

where  $w_2 > w_1 \ge 0$ , and  $\lambda, \alpha \ge 0$  are weight coefficients.

The task rewards  $R_{\rm task}$  follow the previous definitions. Notably, we assign a higher weight  $w_2$  to the final answer  $a_2$  to encourage more accurate reviewed responses while still incentivizing good initial answers. This design also penalizes cases where the first answer is correct but the second is incorrect, pushing the model to improve overall reliability. The term  $R_{\rm fmt}$  ensures that the output format adheres to the required  $answer \rightarrow think \rightarrow answer$  template.

Particularly, the last term  $R_{\text{fallback}} \in \{0,1\}$  is a fallback bonus when  $a_1$  is the designated string "Let's analyze the problem step by step" and  $a_2$  is correct. This discourages low-confidence guesses in  $a_1$  for difficult problems and rewards honest deferral followed by accurate reasoning. It is particularly helpful for math and symbol-heavy questions, where premature guesses are often wrong. Further analysis of the reward design is discussed in Appendix B.

During training, we observe consistent increases in total reward. Notably,  $R_{\text{task}}^{(2)}$  typically exceeds  $R_{\text{task}}^{(1)}$ , confirming the benefit of explicit reasoning for more challenging instances while still retaining fast, correct first answers when appropriate.

