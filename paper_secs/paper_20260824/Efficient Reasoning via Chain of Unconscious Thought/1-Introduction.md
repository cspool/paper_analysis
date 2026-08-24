# 1 Introduction

Large reasoning models (LRMs) [\(OpenAI,](#page-9-0) [2025;](#page-9-0) [Team,](#page-9-1) [2025\)](#page-9-1) have demonstrated promising performance in complex tasks like code, math, and computer use via Chain of Thought (CoT) reasoning [\(Wei et al.,](#page-9-2) [2022\)](#page-9-2). Despite their effectiveness, LRMs are token-inefficient due to high token costs of the reasoning processes [\(Liu et al.,](#page-9-3) [2025b\)](#page-9-3).

To alleviate this problem, the existing trainingbased methods are proposed via supervised finetuning [\(Kang et al.,](#page-8-0) [2024\)](#page-8-0), reinforcement learning [\(Luo et al.,](#page-9-4) [2025\)](#page-9-4), latent CoT [\(Hao et al.,](#page-8-1) [2024\)](#page-8-1), etc. Although effective, they require further training on LRMs, leading to high inference costs. Differently, the more adaptable training-free methods are proposed with prompt engineering strategies [\(Nayab](#page-9-5) [et al.,](#page-9-5) [2024;](#page-9-5) [Xu et al.,](#page-10-0) [2025\)](#page-10-0), reasoning delegation approaches [\(Aytes et al.,](#page-8-2) [2025\)](#page-8-2), and dynamic optimization methods [\(Sui et al.,](#page-9-6) [2025\)](#page-9-6). Despite these advancements, existing methods still generate explicit and redundant reasoning processes, leading to token inefficiency, as shown in Figure [1.](#page-0-0)

To solve this problem, we introduce Unconscious Thought Theory (UTT) from cognitive science, which suggests complex problems can be solved more efficiently through internalized cognitive processes. From this principle, we propose Chain of Unconscious Thought (CoUT), a novel paradigm that encourages models to conduct the reasoning process within their hidden layers. Concretely, it first prompts the model to internalize reasoning processes without emitting detailed chains, thereby achieving significant reasoning compression. In addition, we introduce a bag of tokenefficient strategies to minimize the unnecessary token costs while preserving reasoning accuracy. In this manner, CoUT significantly reduces the explicit token outputs required during inference while maintaining or improving accuracy. To evaluate the effectiveness of CoUT, we conduct exten-

<sup>\*</sup>Equal Contribution.

<span id="page-0-1"></span><sup>1</sup> https://github.com/Rohan-GRH/CoUT

sive experiments on a wide range of mathematical reasoning benchmarks, including both open-ended and multiple-choice questions. These results underscore the potential of leveraging unconscious thought paradigms to enhance efficiency. Extensive experiments demonstrate the effectiveness of CoUT. As shown in Figure [1,](#page-0-0) it notably reduces token usage by 20.51% with only a 0.1% drop in accuracy, outperforming the runner-up on average. The main contributions are summarized as follows.

- We introduce UTT and propose CoUT, a new reasoning paradigm, to improve the token efficiency of LRMs by internalizing the reasoning.
- We design a bag of token-efficient strategies to help models reduce unnecessary tokens while preserving reasoning performance.
- Extensive experiments and analyses demonstrate the effectiveness and efficiency of CoUT.

