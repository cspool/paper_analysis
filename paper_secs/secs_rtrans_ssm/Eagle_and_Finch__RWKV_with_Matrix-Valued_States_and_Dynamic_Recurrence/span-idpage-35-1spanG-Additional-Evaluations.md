# <span id="page-35-1"></span>**G Additional Evaluations**

#### <span id="page-35-2"></span>**G.1 Alignment Benchmark**

Alignment is an important step in creating an assistant LM, because it helps language models generate relevant and helpful responses, as well as avoiding harmful and biased content. Our Eagle models are tested for Chinese alignment using the AlignBench [\(Liu et al.,](#page-20-10) [2023b\)](#page-20-10), a benchmark for evaluating the alignment of Chinese LLMs, featuring 683 diverse and challenging queries across eight categories like language abilities, logical reasoning, and professional knowledge. It

<span id="page-36-2"></span>

| Model <sup>4</sup> | 专业   | 中文   | 基本   | 数学   | 文本   | 综合   | 角色   | 逻辑   | 中文   | 中文   | Total |
|--------------------|------|------|------|------|------|------|------|------|------|------|-------|
|                    | 能力   | 理解   | 任务   | 计算   | 写作   | 问答   | 扮演   | 推理   | 推理   | 语言   |       |
| RWKV-4 7B          | 4.91 | 4.16 | 3.51 | 2.08 | 5.16 | 5.82 | 4.80 | 2.25 | 2.17 | 4.73 | 3.45  |
| Eagle 0.4B         | 2.89 | 2.05 | 2.35 | 1.24 | 3.12 | 3.66 | 2.59 | 1.75 | 1.50 | 2.78 | 2.14  |
| Eagle 1.5B         | 3.87 | 3.02 | 3.18 | 1.63 | 4.33 | 5.34 | 4.06 | 2.23 | 1.93 | 3.97 | 2.95  |
| Eagle 3B           | 4.48 | 3.72 | 3.57 | 2.10 | 4.73 | 5.66 | 4.55 | 2.34 | 2.22 | 4.45 | 3.34  |
| Eagle 7B           | 5.15 | 4.21 | 4.18 | 2.44 | 5.69 | 6.29 | 5.32 | 2.83 | 2.63 | 5.14 | 3.89  |
| Finch 1.6B         | 4.39 | 3.29 | 3.59 | 1.81 | 4.63 | 5.13 | 4.21 | 2.40 | 2.11 | 4.21 | 3.16  |
| Finch 3B           | 4.65 | 3.45 | 3.74 | 2.11 | 4.97 | 5.79 | 5.09 | 2.78 | 2.44 | 4.61 | 3.53  |

Table 14: AlignBench (Liu et al., 2023b), a Chinese benchmark, with header names from left to right: 1) Professional Knowledge, 2) Advanced Chinese Understanding, 3) Fundamental Language Ability, 4) Mathematics, 5) Writing Ability, 6) Open-ended Questions, 7) Task-Oriented Role Play, 8) Logical Reasoning, 9) Reasoning, 10) Chinese. Results Judged by CritiqueLLM (Ke et al., 2023)

employs a rule-calibrated, multi-dimensional LLM-as-Judge methodology with Chain-of-Thought explanations, ensuring high interpretability and reliability.

Table 14 showcases a consistent improvement in the performance of Eagle and Finch models on the AlignBench benchmark as model size and generation progresses. This trend is evident across a wide range of categories, highlighting the larger models' enhanced capability to understand and generate contextually relevant responses. Particularly, both the Eagle 7B and Finch 3B model significantly surpasses its smaller and previous generation counterparts, achieving higher overall scores. This progression underscores the critical role of scaling model size as well as improving architecture in aligning with human judgment in complex language understanding tasks. The results affirm the importance of model architecture and capacity in achieving superior alignment and interpretability in language models.

#### <span id="page-36-0"></span>G.2 MTBench

MTBench (Zheng et al., 2024) evaluates the performance of LLMs in responding to 80 high-quality multi-turn questions. The questions cover 8 common categories of user prompts including writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, and humanities/social science knowledge. Figure 11 shows the results on MTBench. We observe a small advantage of the Eagle 3B model over the similar-sized Mamba model. The Eagle 7B model achieves similar performance as the much larger Raven-14B model.

#### <span id="page-36-1"></span>**G.3** Self-Learning

The Self-Learning process (Ferdinan et al., 2024) allows an LLM to identify its own knowledge gaps and train itself to expand its knowledge. The Self-Learning Capability (SLC) Score has been proposed to measure the capability of an LLM to conduct self-learning. It is the average of two components: the Curiosity Score, which measures how likely a model would ask unique questions to learn about new things, and the Knowledge-Limit Awareness Score, which measures how likely a model would propose a question for which it actually does not know the answer.

We evaluate the self-learning capability of Eagle and compare with existing open models<sup>5</sup>, including RWKV-4 (Peng et al., 2023), neural-chat-7b (Lv et al., 2023), Mistral 7b and 7b-instruct (Jiang et al., 2023), and TinyLlama 1.1B (Zhang et al., 2024). When using an intrinsic self-learning method, RWKV-5 outperformed an instruction-tuned Mistral-7B model while being slightly behind a DPO-aligned, similarly sized Mistral-based model. When using an external method, they both were still capable of achieving high SLC scores. Table 15 shows the full evaluation results, with the top three scores from each method marked in bold.

<span id="page-36-3"></span> $<sup>^5</sup>$ The code is available at https://github.com/teddy-f-47/self-learning-llm-public

<span id="page-37-3"></span>![](_page_37_Figure_0.jpeg)

Figure 11: Comparison of Mamba, RWKV-5 (7B) and RWKV-4 (14B) on MTBench. The Mamba and RWKV-5 models are instruction fine-tuned with the OpenHermes 2.5 dataset. Score generated from GPT-4.

#### <span id="page-37-0"></span>**G.4 Zero-shot evaluation on additional NLP tasks**

Zero-shot evaluation is a difficult setup [\(Sanh et al.,](#page-22-16) [2021;](#page-22-16) [Albalak et al.,](#page-17-14) [2022\)](#page-17-14). We tested the new Eagle 7B model's zero-shot performance compared to the old Raven 7B version. The experiments presented are done on the subsets of datasets also used to test ChatGPT performance in [\(Koco´n](#page-20-12) [et al.,](#page-20-12) [2023\)](#page-20-12). As shown in Table [16,](#page-38-1) the new model consistently outperforms the old one on various tasks. It is to be noted that the new model remains very sensitive to the selected prompt template, just as the old one, as was shown in [\(Peng et al.,](#page-21-0) [2023\)](#page-21-0).

