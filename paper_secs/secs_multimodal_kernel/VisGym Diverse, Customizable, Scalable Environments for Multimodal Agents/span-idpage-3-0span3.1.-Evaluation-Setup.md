# <span id="page-3-0"></span>3.1. Evaluation Setup

We evaluate 12 vision-language models spanning three categories: proprietary (Gemini 3 Pro (team, 2025), Gemini 2.5 Pro (DeepMind, 2025), GPT-5 (OpenAI, 2025), Claude Sonnet 4 (Team, 2025), Grok 4 Fast (xAI, 2025), Qwen-VL-Max (Bai et al., 2025)); open-weight models (Qwen3-VL-235B-Instruct (Yang et al., 2025), GLM-4.5V (Hong et al., 2025), Llama-4-Maverick (Touvron et al., 2023), Qwen-2.5-VL-72B-Instruct (Bai et al., 2025), Gemma 3-27B-Instruct (Team et al., 2025)); and specialized models targeted at GUI/game environments (UI-Tars-1.5-7B (Qin et al., 2025)). We access all proprietary and hosted models through OpenRouter and thus ensure a consistent prompting interface and inference pipeline. We additionally evaluate models that we finetune on solver demonstrations. We provide details of the supervised finetuning setup in Sec. 5.1.

All models are evaluated in a multi-turn manner. At each step t, the model receives the full history

<span id="page-3-3"></span>
$$H_t = (I, \{(o_\tau, a_\tau, f_\tau)\}_{\tau < t}), \tag{1}$$

where  $I \in \mathbb{R}^{L_I}$  is the task instruction,  $o_{\tau} \in \mathbb{R}^{H \times W \times C}$  the observation,  $a_{\tau} \in \mathbb{R}^{L_a}$  the action, and  $f_{\tau} \in \mathbb{R}^{L_f}$  the environment feedback. The model then outputs an action  $a_t$ . If it outputs the stop action, the environment terminates and returns a binary reward indicating task success. In addition, we limit the number of interaction steps to 20 for the easy setting and the tasks of Dot-Pointing and Fetch-Reach, 30 for the hard setting and Fetch Pick-n-Place task. All tasks are designed to be solvable within these limits, and the environment explicitly provides the number of remaining steps as part of its feedback. We also ensure that the length of interaction history is within models' context window. We evaluate each model on 70 episodes per task and setting (i.e., easy, hard).

