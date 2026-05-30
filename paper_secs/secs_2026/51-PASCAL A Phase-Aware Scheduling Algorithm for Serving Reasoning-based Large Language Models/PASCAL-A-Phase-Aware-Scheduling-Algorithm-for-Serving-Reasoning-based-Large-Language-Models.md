# PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

Eunyeong Cho KAIST eunyeong.cho@kaist.ac.kr

Jehyeon Bang KAIST jehyeon.bang@kaist.ac.kr

Ranggi Hwang<sup>∗</sup> UNIST ranggi.hwang@unist.ac.kr

Minsoo Rhu KAIST mrhu@kaist.ac.kr

*Abstract*—The emergence of reasoning-based LLMs leveraging Chain-of-Thought (CoT) inference introduces new serving challenges, as their extended reasoning phases delay user-visible output and inflate Time-To-First-Token (TTFT). Existing LLM serving frameworks fail to distinguish between reasoning and answering phases, leading to performance degradation under GPU memory constraints. We present PASCAL, a phase-aware scheduling algorithm that prioritizes reasoning to reduce TTFT while using controlled preemption and token pacing during answering to preserve Quality-of-Experience (QoE). Our hierarchical scheduler combines instance-level placement with intrainstance execution and enables dynamic migration at phase boundaries to balance load and reduce interference. Across benchmarks using DeepSeek-R1-Distill-Qwen-32B, PASCAL reduces tail TTFT by up to 72% while maintaining answering phase SLO attainment, demonstrating the importance of phaseaware scheduling for reasoning-based LLM deployment.

*Index Terms*—Serving system, reasoning-based LLMs, scheduling framework, user experience.

