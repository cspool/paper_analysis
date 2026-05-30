# Supplementary Material

## Outline

This Supplementary Material complements the main paper, providing comprehensive experimental details, in-depth analyses of training dynamics, and extensive qualitative visualizations. The content is organized as follows:

- Strategic Alignment & Motivation. We first demonstrate the conceptual alignment between LongVT and state-ofthe-art proprietary large multimodal models (LMMs) in Section [1.](#page-1-0) Subsequently, we present a rigorous data contamination study in Section [2](#page-2-1) to underscore the necessity of our proposed VideoSIAH-Eval benchmark, followed by detailed statistics of the curated dataset in Section [3.](#page-2-2)
- Formulation & Training Dynamics. We present the overall framework illustration in Figure [3](#page-14-0) and elaborate on the theoretical formulations of our training objectives in Section [4](#page-4-1) for both supervised fine-tuning (SFT) and reinforcement learning (RL). Crucially, in Section [5,](#page-5-1) we visualize the "economy of thinking"—a distinct evolutionary trajectory where the model learns to internalize tool usage. Section [6](#page-7-1) then provides the exact hyperparameters and infrastructure details for reproducibility.
- Efficiency & Qualitative Analysis. We report a detailed inference latency comparison in Section [7,](#page-15-0) countering the intuition that multi-turn agentic frameworks are inherently slower. In Section [8,](#page-16-0) we provide prompt templates, diverse qualitative examples, and workflow demonstration, while Section [9](#page-16-1) analyzes specific failure modes to highlight the importance of the cold-start training stage.
- Discussion. Finally, we discuss the architectural limitations and future multi-agent directions in Section [10,](#page-16-2) followed by a discussion on the broader impact and ethical considerations in Section [11](#page-16-3) and Section [12,](#page-17-0) respectively.

