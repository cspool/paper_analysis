# C Prompt Template

In the main paper, we introduce the system prompt used in VideoAuto-R1, which adopts an answer → think → answer format. This prompt design avoids a cold-start stage and facilitates stable training with promising performance. Additionally, in Table 5 of the main paper, we explore alternative reinforcement learning settings.

RL without Thinking. As shown in Table [13,](#page-20-3) this variant directly applies GRPO without requiring any intermediate explanation. The model is prompted to provide only the final answer enclosed in a \\boxed{} command.

RL with Thinking. As shown in Table [14,](#page-20-4) this is the standard prompt for GRPO training. The model first generates a reasoning trace within <think> </think> tags, followed by the final answer enclosed in \\boxed{}. This prompt format aligns with previous R1-style approaches such as Video-R1 [\(Feng et al.,](#page-13-2) [2025\)](#page-13-2) and VideoChat-R1 [\(Li et al.,](#page-14-3) [2025c\)](#page-14-3).

<span id="page-20-3"></span>Table 13 System Prompt for RL without Thinking.

#### SYSTEM PROMPT

You are a helpful assistant. Put your final answer in \\boxed{}.

#### <span id="page-20-4"></span>Table 14 System Prompt for RL with Thinking.

### SYSTEM PROMPT

You are a helpful assistant.

<span id="page-20-1"></span>FIRST, think through the reasoning process as an internal monologue, and THEN provide the final answer. The reasoning process MUST be enclosed within <think> </think> tags, and the final answer MUST be wrapped in \\boxed{}.

<span id="page-21-2"></span>![](_page_21_Figure_0.jpeg)

Figure 6 Training Curves of VideoAuto-R1. We show the average task reward for both initial and reviewed answers during GRPO training.

