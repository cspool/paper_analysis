# <span id="page-7-0"></span>**5.1. Supervised Fine-Tuning Experiments**

We generate demonstration trajectories for supervised fine-tuning using the multi-step solver described in Sec. 2. We apply two preprocessing filters: (1) discarding trajectories that fail to complete the task, and (2) removing trajectories with initial states overlapping the test split to prevent data leakage.

We evaluate two fine-tuning configurations: single-task and mixed-task. In the single-task setting, we fine-tune a separate model for each task, whereas in the *mixed-task* setting, a single model is trained jointly on all tasks. Notably, demonstrations are sourced exclusively from the easy difficulty level; thus, performance on the hard setting serves as a metric for difficulty generalization. All experiments employ Qwen2.5-VL-7B-Instruct (Bai et al., 2025) with full-parameter fine-tuning, a global batch size of 64, a learning rate of  $1 \times 10^{-5}$ , and bf16 precision. Models are trained for 1,500 steps in the single-task setting and 5,000 steps in the mixed-task setting. We utilize LlamaFactory (Zheng et al., 2024) for all data preprocessing and training orchestration.

**Results.** As shown in Figs. 2 and 4, finetuned models achieve state-of-the-art performance on most tasks, validating both the learnability of our environments and the effectiveness of our multi-step solvers. These gains confirm that current VLMs can substantially benefit from structured, solver-generated demonstrations in visually grounded multi-step settings.

