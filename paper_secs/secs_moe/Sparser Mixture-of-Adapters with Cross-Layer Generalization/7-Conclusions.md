# 7 Conclusions

We introduce SMOA, a new approach designed to address the limitations of traditional MoA methods in fine-tuning LLMs. SMOA leverages a unified cross-layer shared pool of adapters, enabling a significantly sparser selection of experts for each instance while maintaining robust performance. We further enhance specialization by directing experts to focus on residual information not covered by the backbone. Extensive evaluations across various LLMs consistently show that SMOA outperforms existing methods in both in-distribution and outof-distribution tasks, achieving superior generalization with fewer activated adapters. These results position SMOA as a step forward in developing more adaptable MoA frameworks for LLMs.

