# <span id="page-22-0"></span>A.9 Qwen2.5-Math-1.5B Training Details

In this paper, we used two models as base models: Qwen2.5-1.5B-Math and Qwen2.5-3B. For the RLVR-trained version of the 1.5B model, we used Qwen2.5-Math-1.5B-Oat-Zero[2](#page-22-2) , a publicly available model trained by [Liu et al..](#page-9-11) According to their report, the model was trained with Dr.GRPO [\(Liu et al.,](#page-9-11) [2025b\)](#page-9-11), a variant of the GRPO algorithm [\(Shao et al.,](#page-10-13) [2024\)](#page-10-13) designed to remove response length and question difficulty biases. The model was trained on questions from level 3 to 5 from the MATH training set. For the 3B model, we performed RLVR training ourselves. Training details are shown right below in Appendix [A.10](#page-22-1)

## <span id="page-22-1"></span>A.10 Qwen2.5-3B RLVR Training Details

For RLVR training of Qwen2.5-3B, we used the GRPOTrainer from the TRL[3](#page-22-3) library, which implements the standard GRPO algorithm. The model was trained on the full MATH training set, consisting of 7,500 questions.

## A.10.1 Prompt Setting

Prior work has shown that the performance of smaller models can be sensitive to prompt design [\(Hochlehn](#page-9-19)[ert et al.,](#page-9-19) [2025;](#page-9-19) [Liu et al.,](#page-9-11) [2025b\)](#page-9-11). Following [Liu et al.,](#page-9-11) we evaluated three prompt formats, as listed below. We ultimately adopted Template 3 (question only), which yielded the best performance.

## Prompt Templates

Template 1 (R1 template) A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer> tags. User: {question} Assistant: <think> reasoning here </think> <answer> answer here </answer>

Template 2 (Qwen-Math template) <|im start|>system Please reason step by step, and put your final answer within \boxed{}. <|im end|> <|im start|>user {question} <|im end|> <|im start|>assistant

Template 3 (Question only) {question}

