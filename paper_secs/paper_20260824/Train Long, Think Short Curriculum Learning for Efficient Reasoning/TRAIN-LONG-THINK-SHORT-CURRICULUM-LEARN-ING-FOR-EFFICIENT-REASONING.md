# TRAIN LONG, THINK SHORT: CURRICULUM LEARN-ING FOR EFFICIENT REASONING

 $\begin{array}{llllllllllllllllllllllllllllllllllll$ 

#### **ABSTRACT**

Recent work on enhancing the reasoning abilities of large language models (LLMs) has introduced explicit length control as a means of constraining computational cost while preserving accuracy. However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning. In this work, we propose a curriculum learning strategy for length-controlled reasoning using Group Relative Policy Optimization (GRPO). Our method starts with generous token budgets and gradually tightens them over training, encouraging models to first discover effective solution strategies and then distill them into more concise reasoning traces. We augment GRPO with a reward function that balances three signals: task correctness (via verifier feedback), length efficiency, and formatting adherence (via structural tags). Experiments on GSM8K, MATH500, SVAMP, College Math, and GSM+ demonstrate that curriculum-based training consistently outperforms fixed-budget baselines at the same final budget, achieving higher accuracy and significantly improved token efficiency. We further ablate the impact of reward weighting and decay schedule design, showing that progressive constraint serves as a powerful inductive bias for training efficient reasoning models. Our code and checkpoints are released at: https://github.com/hammoudhasan/ curriculum\_grpo.

### 1 Introduction

Recent advances in large language models (LLMs) have enabled impressive capabilities across a wide range of natural language processing tasks. A key challenge now is equipping these models with robust *reasoning* abilities, enabling them to solve problems that require systematic, multi-step inference.

To date, two main paradigms have emerged to improve reasoning in LLMs. The first relies on *supervised fine-tuning* (SFT) on datasets containing *chain-of-thought* (CoT) annotations, where human experts provide intermediate reasoning steps. Although SFT is straightforward to implement, it depends on costly data collection and may struggle to generalize beyond seen distributions. The second paradigm uses *reinforcement learning* (RL) to directly optimize the behavior of the model through feedback on the completed reasoning traces. RL-based methods avoid explicit reasoning annotations, can leverage sparse rewards, and have achieved state-of-the-art performance in recent systems.

Within the RL category, *Group Relative Policy Optimization* (GRPO) has shown particular promise. GRPO fine-tunes LLMs without a separate value function by sampling a group of candidate responses per prompt and normalizing rewards across that group. This group-relative normalization stabilizes learning from sparse correctness signals and encourages the model to prefer responses that are strong relative to its own cohort.

<sup>&</sup>lt;sup>1</sup>King Abdullah University of Science and Technology (KAUST), Saudi Arabia

<sup>&</sup>lt;sup>2</sup>Massachusetts Institute of Technology (MIT), Cambridge, MA, USA

<sup>&</sup>lt;sup>3</sup>Princeton University, Princeton, NJ, USA

<sup>\*</sup>Correspondence to: hasanabedalkader.hammoud@kaust.edu.sa

An orthogonal line of work incorporates explicit *length control* into reasoning training: models are trained to produce reasoning traces under token-budget constraints, balancing solution quality and efficiency. Prior methods that handle multiple fixed budgets independently fail to leverage the natural progression of capability that can arise if the model is first allowed longer reasoning chains and then gradually required to compress them.

In this paper, we introduce curriculum learning for length-controlled reasoning. Instead of fixing the budget throughout the training, we begin with a large initial token budget B<sup>0</sup> and progressively tighten it via an exponential decay schedule:

$$B(t) = \max\left(1, B_0 \cdot \gamma^{\left\lfloor \frac{t}{T} \right\rfloor}\right),$$

where γ ∈ (0, 1) is the decay factor and T is the step interval between budget updates. During training, the model can explore a long chain-of-thought to discover effective reasoning patterns; as the budget shrinks, it is forced to distill these patterns into more concise and efficient reasoning traces.

We train with GRPO-based curriculum length control on two complementary mathematical reasoning datasets: GSM8K and MATH500. We then evaluate zero-shot performance on GSM8K, MATH500, SVAMP, College Math, and GSM+, comparing against fixed-budget GRPO baselines and base models without reasoning fine-tuning. Our experiments, conducted with QWEN-2.5-7B, show that curriculum learning yields consistent gains in both accuracy and token efficiency at the same final budget, indicating that progressive constraint is a powerful inductive signal for efficient reasoning.

Our contributions are as follows.

- 1. We propose a curriculum learning strategy for length-controlled reasoning by embedding an exponentially decaying token budget into GRPO fine-tuning, enabling a smooth transition from exploration to compression of reasoning chains.
- 2. We empirically demonstrate that curriculum-based length control outperforms fixed-budget training across multiple benchmarks, improving reasoning accuracy while reducing average token usage.
- 3. We release a reproducible implementation built on torchtune along with pretrained checkpoints to accelerate future work on LLMs capable of efficient reasoning.

