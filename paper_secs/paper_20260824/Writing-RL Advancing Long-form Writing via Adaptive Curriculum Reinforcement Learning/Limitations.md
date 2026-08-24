# Limitations

Here we discuss several limitations of this work. To scale up model size. While the performance gain by training writer models at 7B and 14B scales is relatively large, there remains considerable room for exploration at larger model scales. Prior research has shown that the underlying capability of the base model plays a crucial role in the effectiveness of RL [\(Gandhi et al.,](#page-9-11) [2025\)](#page-9-11). Therefore, applying Writing-RL to stronger models may lead to even greater performance improvements, as well as more pronounced generalization effects from long-output generation to long-input reasoning.

To explore the *zero phenomenon* of RL. This work demonstrates that reinforcement learning, when applied to long-form generation, can elicit strong performance gains and even induce generalization to long-input reasoning. While an intriguing research direction is to investigate this phenomenon from a more fundamental perspective by directly applying RL to base models without prior supervised fine-tuning. Such a setup may offer clearer insight into whether RL alone is sufficient to induce strong long-form generation capabilities.

