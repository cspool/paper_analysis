# <span id="page-15-1"></span>**4. Alignment**

#### <span id="page-15-2"></span>**4.1. Supervised Fine-Tuning**

<span id="page-15-3"></span>Building upon our prior research [\(DeepSeek-AI, 2024\)](#page-22-1), we curate our instruction tuning datasets to include 1.5M instances, comprising 1.2M instances for helpfulness and 0.3M instances for safety. In comparison to the initial version, we improve the data quality to mitigate hallucinatory responses and enhance writing proficiency. We fine-tune DeepSeek-V2 with 2 epochs, and the learning rate is set to 5 × 10−<sup>6</sup> . For the evaluation of DeepSeek-V2 Chat (SFT), we mainly include generation-based benchmarks, except for several representative multiple-choice tasks (MMLU and ARC). We also conduct an instruction-following evaluation (IFEval) [\(Zhou et al.,](#page-25-5) [2023\)](#page-25-5) for DeepSeek-V2 Chat (SFT), using prompt-level loose accuracy as the metric. Moreover, we employ LiveCodeBench [\(Jain et al., 2024\)](#page-22-13) questions from September 1st, 2023 to April 1st, 2024 to evaluate chat models. In addition to the standard benchmarks, we further evaluate our model on open-ended conversation benchmarks including MT-Bench [\(Zheng et al., 2023\)](#page-25-0), AlpacaEval 2.0 [\(Dubois et al., 2024\)](#page-22-2), and AlignBench [\(Liu et al., 2023\)](#page-23-3). For comparison, we also evaluate Qwen1.5 72B Chat, LLaMA-3-70B Instruct, and Mistral-8x22B Instruct in our evaluation framework and settings. As for DeepSeek 67B Chat, we directly refer to the evaluation results reported in our previous release.

## **4.2. Reinforcement Learning**

In order to further unlock the potential of DeepSeek-V2 and align it with human preference, we conduct Reinforcement Learning (RL) to adjust its preference.

**Reinforcement Learning Algorithm.** In order to save the training costs of RL, we adopt Group Relative Policy Optimization (GRPO) [\(Shao et al., 2024\)](#page-24-3), which foregoes the critic model that is typically with the same size as the policy model, and estimates the baseline from group scores instead. Specifically, for each question , GRPO samples a group of outputs {1, 2, · · · , } from the old policy and then optimizes the policy model by maximizing the following objective:

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[q \sim P(Q), \left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right]$$

$$\frac{1}{G} \sum_{i=1}^{G} \left(\min\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}A_{i}, \operatorname{clip}\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}, 1 - \varepsilon, 1 + \varepsilon\right)A_{i}\right) - \beta \mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)\right), \tag{32}$$

$$\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right) = \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - \log\frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1,\tag{33}$$

where and are hyper-parameters; and is the advantage, computed using a group of rewards {1,2, . . . ,} corresponding to the outputs within each group:

$$A_{i} = \frac{r_{i} - \text{mean}(\{r_{1}, r_{2}, \cdots, r_{G}\})}{\text{std}(\{r_{1}, r_{2}, \cdots, r_{G}\})}.$$
(34)

**Training Strategy.** In our preliminary experiments, we find that the RL training on reasoning data, such as code and math prompts, exhibits unique characteristics that are distinct from the training on general data. For example, the mathematical and coding abilities of our model can keep improving over a longer period of training steps. Therefore, we employ a two-stage RL training strategy, which first performs reasoning alignment, and then performs human preference alignment. In the first reasoning alignment stage, we train a reward model for code and math reasoning tasks, and optimize the policy model with the feedback of :

$$r_i = RM_{reasoning}(o_i). (35)$$

In the second human preference alignment stage, we adopt a multi-reward framework, which acquires rewards from a helpful reward model ℎ , a safety reward model , and a rule-based reward model . The final reward of a response is

$$r_i = c_1 \cdot RM_{helpful}(o_i) + c_2 \cdot RM_{safety}(o_i) + c_3 \cdot RM_{rule}(o_i), \tag{36}$$

where 1, 2, and <sup>3</sup> are corresponding coefficients.

In order to obtain reliable reward models that play crucial roles in the RL training, we carefully collect preference data, and meticulously conduct quality filtering and proportion adjustments. We obtain code preference data based on compiler-feedback, and mathematical preference data based on the ground-truth labels. For reward model training, we initialize the reward models with DeepSeek-V2 Chat (SFT) and train them with either a point-wise or a pair-wise loss. In our experiments, we observe that the RL training can fully tap into and activate the potential of our model, enabling it to select the correct and satisfactory answer from possible responses.

**Optimizations for Training Efficiency.** Conducting RL training on extremely large models places high demands on the training framework. It requires careful engineering optimization to manage the GPU memory and RAM pressure, and meanwhile maintain a fast training speed. For this goal, we implement the following engineering optimizations. (1) Firstly, we propose a hybrid engine that adopts different parallel strategies for training and inference respectively to achieve higher GPU utilization. (2) Secondly, we leverage vLLM [\(Kwon et al., 2023\)](#page-23-11) with large batch sizes as our inference backend to accelerate the inference speed. (3) Thirdly, we carefully design a scheduling strategy for offloading models to CPUs and loading models back to GPUs, which achieves a near-optimal balance between the training speed and memory consumption.

## <span id="page-17-0"></span>**4.3. Evaluation Results**

**Evaluations on Standard Benchmarks.** Initially, we evaluate DeepSeek-V2 Chat (SFT) and DeepSeek-V2 Chat (RL) on standard benchmarks. Notably, DeepSeek-V2 Chat (SFT) demonstrates substantial improvements in GSM8K, MATH, and HumanEval evaluations compared with its base version. This progress can be attributed to the inclusion of our SFT data, which comprises a considerable volume of math and code related content. In addition, DeepSeek-V2 Chat (RL) further boosts the performance on math and code benchmarks. We show more code and math evaluations in Appendix [F.](#page-31-0)

As for the comparisons with other models, we first compare DeepSeek-V2 Chat (SFT) with Qwen1.5 72B Chat, and find that DeepSeek-V2 Chat (SFT) surpasses Qwen1.5 72B Chat on almost all of English, math, and code benchmarks. On Chinese benchmarks, DeepSeek-V2 Chat (SFT) demonstrates slightly lower scores than Qwen1.5 72B Chat on multi-subject multiple-choice tasks, consistent with the performance observed from their base versions. When compared with the state-of-the-art open-source MoE model, Mixtral 8x22B Instruct, DeepSeek-V2 Chat (SFT) exhibits better performance on most benchmarks, except for NaturalQuestions and IFEval. Furthermore, in comparison to the state-of-the-art open-source model LLaMA3 70B Chat, DeepSeek-V2 Chat (SFT) shows similar performance in code and math related benchmarks. LLaMA3 70B Chat exhibits better performance on MMLU and IFEval, while DeepSeek-V2 Chat (SFT) showcases stronger performance on Chinese tasks. Ultimately, DeepSeek-V2 Chat (RL) demonstrates further enhanced performance in both mathematical and coding tasks compared with DeepSeek-V2 Chat (SFT). These comparisons highlight the strengths of DeepSeek-V2 Chat in relation to other language models in various domains and languages.

**Evaluations on Open-Ended Generation.** We proceed with additional evaluations of our models on open-ended conversation benchmarks. For English open-ended conversation generation, we utilize MT-Bench and AlpacaEval 2.0 as the benchmarks. Evaluation results presented in Table [4](#page-18-1) demonstrate a significant performance advantage of DeepSeek-V2 Chat (RL) over DeepSeek-V2 Chat (SFT). This outcome showcases the effectiveness of our RL training in achieving improved alignment. In comparison to other open-source models, DeepSeek-V2 Chat (RL) demonstrates superior performance over Mistral 8x22B Instruct and Qwen1.5 72B Chat on both benchmarks. When compared with LLaMA3 70B Instruct, DeepSeek-V2 Chat (RL) showcases competitive performance on MT-Bench and notably outperforms it on AlpacaEval 2.0. These results highlight the strong performance of DeepSeek-V2 Chat (RL) in generating high-quality and contextually relevant responses, particularly in instruction-based conversation tasks.

In addition, we evaluate the Chinese open-ended generation capability based on AlignBench. As presented in Table [5,](#page-19-0) DeepSeek-V2 Chat (RL) exhibits a slight advantage over DeepSeek-V2 Chat (SFT). Notably, DeepSeek-V2 Chat (SFT) surpasses all open-source Chinese models by a significant margin. It significantly outperforms the second-best open-source model, Qwen1.5

|         | Benchmark             |         | # Shots DeepSeek Qwen 1.5 LLaMA3<br>67B Chat |       |       | Mixtral<br>72B Chat 70B Inst. 8x22B Inst. | DeepSeek-V2 DeepSeek-V2<br>Chat (SFT) | Chat (RL) |
|---------|-----------------------|---------|----------------------------------------------|-------|-------|-------------------------------------------|---------------------------------------|-----------|
|         | Context Length        | -       | 4K                                           | 32K   | 8K    | 64K                                       | 128K                                  | 128K      |
|         | Architecture          | -       | Dense                                        | Dense | Dense | MoE                                       | MoE                                   | MoE       |
|         | # Activated Params -  |         | 67B                                          | 72B   | 70B   | 39B                                       | 21B                                   | 21B       |
|         | # Total Params        | -       | 67B                                          | 72B   | 70B   | 141B                                      | 236B                                  | 236B      |
|         | TriviaQA              | 5-shot  | 81.5                                         | 79.6  | 69.1  | 80.0                                      | 85.4                                  | 86.7      |
|         | NaturalQuestions      | 5-shot  | 47.0                                         | 46.9  | 44.6  | 54.9                                      | 51.9                                  | 53.4      |
|         | MMLU                  | 5-shot  | 71.1                                         | 76.2  | 80.3  | 77.8                                      | 78.4                                  | 77.8      |
|         | ARC-Easy              | 25-shot | 96.6                                         | 96.8  | 96.9  | 97.1                                      | 97.6                                  | 98.1      |
| English | ARC-Challenge         | 25-shot | 88.9                                         | 91.7  | 92.6  | 90.0                                      | 92.5                                  | 92.3      |
|         | BBH                   | 3-shot  | 71.7                                         | 65.9  | 80.1  | 78.4                                      | 81.3                                  | 79.7      |
|         | AGIEval               | 0-shot  | 46.4                                         | 62.8  | 56.6  | 41.4                                      | 63.2                                  | 61.4      |
|         | IFEval                | 0-shot  | 55.5                                         | 57.3  | 79.7  | 72.1                                      | 64.1                                  | 63.8      |
|         | HumanEval             | 0-shot  | 73.8                                         | 68.9  | 76.2  | 75.0                                      | 76.8                                  | 81.1      |
|         | MBPP                  | 3-shot  | 61.4                                         | 52.2  | 69.8  | 64.4                                      | 70.4                                  | 72.0      |
| Code    | CRUXEval-I-COT        | 2-shot  | 49.1                                         | 51.4  | 61.1  | 59.4                                      | 59.5                                  | 61.5      |
|         | CRUXEval-O-COT 2-shot |         | 50.9                                         | 56.5  | 63.6  | 63.6                                      | 60.7                                  | 63.0      |
|         | LiveCodeBench         | 0-shot  | 18.3                                         | 18.8  | 30.5  | 25.0                                      | 28.7                                  | 32.5      |
|         | GSM8K                 | 8-shot  | 84.1                                         | 81.9  | 93.2  | 87.9                                      | 90.8                                  | 92.2      |
|         | MATH                  | 4-shot  | 32.6                                         | 40.6  | 48.5  | 49.8                                      | 52.7                                  | 53.9      |
| Math    | CMath                 | 0-shot  | 80.3                                         | 82.8  | 79.2  | 75.1                                      | 82.0                                  | 81.9      |
|         | CLUEWSC               | 5-shot  | 78.5                                         | 90.1  | 85.4  | 75.8                                      | 88.6                                  | 89.9      |
| Chinese | C-Eval                | 5-shot  | 65.2                                         | 82.2  | 67.9  | 60.0                                      | 80.9                                  | 78.0      |
|         | CMMLU                 | 5-shot  | 67.8                                         | 82.9  | 70.7  | 61.0                                      | 82.4                                  | 81.6      |

<span id="page-18-1"></span>Table 3 | Comparison among DeepSeek-V2 Chat (SFT), DeepSeek-V2 Chat (RL), and other representative open-source chat models. Regarding TriviaQA and NaturalQuestions, it is worth noting that chat models, such as LLaMA3 70B Instruct, might not strictly adhere to the format constraints typically specified in the few-shot setting. Consequently, this can lead to underestimation of certain models in our evaluation framework.

| Model                       | MT-Bench | AlpacaEval 2.0 |
|-----------------------------|----------|----------------|
| DeepSeek 67B Chat           | 8.35     | 16.6           |
| Mistral 8x22B Instruct v0.1 | 8.66     | 30.9           |
| Qwen1.5 72B Chat            | 8.61     | 36.6           |
| LLaMA3 70B Instruct         | 8.95     | 34.4           |
| DeepSeek-V2 Chat (SFT)      | 8.62     | 30.0           |
| DeepSeek-V2 Chat (RL)       | 8.97     | 38.9           |

Table 4 | English open-ended conversation evaluations. For AlpacaEval 2.0, we use the lengthcontrolled win rate as the metric.

<span id="page-18-0"></span>72B Chat on both Chinese reasoning and language. Moreover, both DeepSeek-V2 Chat (SFT) and DeepSeek-V2 Chat (RL) outperform GPT-4-0613 and ERNIEBot 4.0, solidifying the position of our models in the top-tier LLMs that support Chinese. Specifically, DeepSeek-V2 Chat (RL) shows remarkable performance in Chinese language understanding, which outperforms all models including GPT-4-Turbo-1106-Preview. On the other hand, the reasoning capability of DeepSeek-V2 Chat (RL) still lags behind giant models, such as Erniebot-4.0 and GPT-4s.

<span id="page-19-0"></span>

| Model                          | Overall | Reaso            | oning 中           | 文推理               |                  |                   | Langu            | iage 中文           | 语言                |                   |                   |
|--------------------------------|---------|------------------|-------------------|-------------------|------------------|-------------------|------------------|-------------------|-------------------|-------------------|-------------------|
| 模型                             | 总分      | Avg.<br>推理<br>总分 | Math.<br>数学<br>计算 | Logi.<br>逻辑<br>推理 | Avg.<br>语言<br>总分 | Fund.<br>基本<br>任务 | Chi.<br>中文<br>理解 | Open.<br>综合<br>问答 | Writ.<br>文本<br>写作 | Role.<br>角色<br>扮演 | <b>Pro.</b> 专业 能力 |
| GPT-4-1106-Preview             | 8.01    | 7.73             | 7.80              | 7.66              | 8.29             | 7.99              | 7.33             | 8.61              | 8.67              | 8.47              | 8.65              |
| DeepSeek-V2 Chat (RL)          | 7.91    | 7.45             | 7.77              | 7.14              | 8.36             | 8.10              | 8.28             | 8.37              | 8.53              | 8.33              | 8.53              |
| ERNIEBot-4.0-202404* (文心一言)    | 7.89    | 7.61             | 7.81              | 7.41              | 8.17             | 7.56              | 8.53             | 8.13              | 8.45              | 8.24              | 8.09              |
| DeepSeek-V2 Chat (SFT)         | 7.74    | 7.30             | 7.34              | 7.26              | 8.17             | 8.04              | 8.26             | 8.13              | 8.00              | 8.10              | 8.49              |
| GPT-4-0613                     | 7.53    | 7.47             | 7.56              | 7.37              | 7.59             | 7.81              | 6.93             | 7.42              | 7.93              | 7.51              | 7.94              |
| ERNIEBot-4.0-202312* (文心一言)    | 7.36    | 6.84             | 7.00              | 6.67              | 7.88             | 7.47              | 7.88             | 8.05              | 8.19              | 7.84              | 7.85              |
| Moonshot-v1-32k-202404* (月之暗面) | 7.22    | 6.42             | 6.41              | 6.43              | 8.02             | 7.82              | 7.58             | 8.00              | 8.22              | 8.19              | 8.29              |
| Qwen1.5-72B-Chat*              | 7.19    | 6.45             | 6.58              | 6.31              | 7.93             | 7.38              | 7.77             | 8.15              | 8.02              | 8.05              | 8.24              |
| DeepSeek-67B-Chat              | 6.43    | 5.75             | 5.71              | 5.79              | 7.11             | 7.12              | 6.52             | 7.58              | 7.20              | 6.91              | 7.37              |
| ChatGLM-Turbo(智谱清言)            | 6.24    | 5.00             | 4.74              | 5.26              | 7.49             | 6.82              | 7.17             | 8.16              | 7.77              | 7.76              | 7.24              |
| ERNIEBot-3.5(文心一言)             | 6.14    | 5.15             | 5.03              | 5.27              | 7.13             | 6.62              | 7.60             | 7.26              | 7.56              | 6.83              | 6.90              |
| Yi-34B-Chat*                   | 6.12    | 4.86             | 4.97              | 4.74              | 7.38             | 6.72              | 7.28             | 7.76              | 7.44              | 7.58              | 7.53              |
| GPT-3.5-Turbo-0613             | 6.08    | 5.35             | 5.68              | 5.02              | 6.82             | 6.71              | 5.81             | 7.29              | 7.03              | 7.28              | 6.77              |
| ChatGLM-Pro(智谱清言)              | 5.83    | 4.65             | 4.54              | 4.75              | 7.01             | 6.51              | 6.76             | 7.47              | 7.07              | 7.34              | 6.89              |
| SparkDesk-V2(讯飞星火)             | 5.74    | 4.73             | 4.71              | 4.74              | 6.76             | 5.84              | 6.97             | 7.29              | 7.18              | 6.92              | 6.34              |
| Qwen-14B-Chat                  | 5.72    | 4.81             | 4.91              | 4.71              | 6.63             | 6.90              | 6.36             | 6.74              | 6.64              | 6.59              | 6.56              |
| Baichuan2-13B-Chat             | 5.25    | 3.92             | 3.76              | 4.07              | 6.59             | 6.22              | 6.05             | 7.11              | 6.97              | 6.75              | 6.43              |
| ChatGLM3-6B                    | 4.97    | 3.85             | 3.55              | 4.14              | 6.10             | 5.75              | 5.29             | 6.71              | 6.83              | 6.28              | 5.73              |
| Baichuan2-7B-Chat              | 4.97    | 3.66             | 3.56              | 3.75              | 6.28             | 5.81              | 5.50             | 7.13              | 6.84              | 6.53              | 5.84              |
| InternLM-20B                   | 4.96    | 3.66             | 3.39              | 3.92              | 6.26             | 5.96              | 5.50             | 7.18              | 6.19              | 6.49              | 6.22              |
| Qwen-7B-Chat                   | 4.91    | 3.73             | 3.62              | 3.83              | 6.09             | 6.40              | 5.74             | 6.26              | 6.31              | 6.19              | 5.66              |
| ChatGLM2-6B                    | 4.48    | 3.39             | 3.16              | 3.61              | 5.58             | 4.91              | 4.52             | 6.66              | 6.25              | 6.08              | 5.08              |
| InternLM-Chat-7B               | 3.65    | 2.56             | 2.45              | 2.66              | 4.75             | 4.34              | 4.09             | 5.82              | 4.89              | 5.32              | 4.06              |
| Chinese-LLaMA-2-7B-Chat        | 3.57    | 2.68             | 2.29              | 3.07              | 4.46             | 4.31              | 4.26             | 4.50              | 4.63              | 4.91              | 4.13              |
| LLaMA-2-13B-Chinese-Chat       | 3.35    | 2.47             | 2.21              | 2.73              | 4.23             | 4.13              | 3.31             | 4.79              | 3.93              | 4.53              | 4.71              |

Table 5 | AlignBench leaderboard rated by GPT-4-0613. Models are ranked in descending order based on the overall score. Models marked with \* represent that we evaluate them through their API service or open-weighted model, instead of referring to the results reported in their original papers. Suffixes of Erniebot-4.0 and Moonshot denote the timestamps when we called their API.

## 4.4. Discussion

Amount of SFT Data. The discussion surrounding the necessity of a large SFT corpus has been a topic of intense debate. Previous works (Young et al., 2024; Zhou et al., 2024) argue that fewer than 10K instances of SFT data are enough to produce satisfactory results. However, in our experiments, we observe a significant performance decline on the IFEval benchmark if we use fewer than 10K instances. A possible explanation is that, a language model necessitates a certain amount of data to develop specific skills. Although the requisite data amount may diminish with the model size increasing, it cannot be entirely eliminated. Our observation underscores the critical need for sufficient data to equip an LLM with desired capabilities. Moreover, the quality of SFT data is also crucial, especially for tasks involving writing or open-ended questions.

Alignment Tax of Reinforcement Learning. During human preference alignment, we observe a significant performance enhancement on the open-ended generation benchmarks, in terms of the scores rated by both AI and human evaluators. However, we also notice a phenomenon of "alignment tax" (Ouyang et al., 2022), i.e., the alignment process can negatively impact the performance on some standard benchmarks such as BBH. In order to alleviate the alignment tax, during the RL stage, we make significant efforts in data processing and improving training strategies, finally achieving a tolerable trade-off between the performance on standard and open-ended benchmarks. Exploring how to align a model with human preferences without

compromising its general performance presents a valuable direction for future research.

**Online Reinforcement Learning.** In our preference alignment experiments, we find that the online approach significantly outperforms the offline approach. Therefore, we invest tremendous efforts in implementing an online RL framework for aligning DeepSeek-V2. The conclusion about online or offline preference alignment can vary in different contexts, and we reserve a more thorough comparison and analysis between them for future work.

