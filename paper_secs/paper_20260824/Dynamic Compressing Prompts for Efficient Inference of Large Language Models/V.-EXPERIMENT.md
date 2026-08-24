# V. EXPERIMENT

<span id="page-5-0"></span>In this section, we first introduce the experimental settings in subsection V-A. Our proposed LLM-DCP is compared against the state-of-the-art (SOTA) prompt compression methods in subsection V-B. We show relevant examples of the proposed LLM-DCP in subsection V-C. We also performed numerous ablation experiments to validate the effectiveness of LLM-DCP and to gain a deeper understanding of the proposed method in subsection V-D. Additionally, we further discuss the effect of different hyperparameters on the proposed method in subsection V-E.

## <span id="page-5-4"></span>A. Experimental Settings

- 1) Compared methods: Following the previous working setup [12], we compare the proposed LLM-DCP with only three SOTA task-agnostic prompt compression methods.
  - Selective-Context [20]: Use a small model to compute the self-information of each token and fuse it into a lexical unit u (each lexical unit consists of multiple tokens  $(x_t, ..., x_{t+\alpha})$ ), retaining lexical unit self-information over a threshold value.
  - LLMLingua [18]: It dynamically assigns different compression ratios (τ, τ<sub>que</sub>, τ<sub>ins</sub>, τ<sub>dems</sub>) to the various parts of the prompt, divide the prompt into multiple segments S = {s<sub>1</sub>, s<sub>2</sub>,...,s<sub>m</sub>}, where tokens greater than threshold in each segment are retained.

<span id="page-6-6"></span>

| Method                 | Pub.'Year  | 1-shot constraint |            |                   | half-shot constraint |            |                   |
|------------------------|------------|-------------------|------------|-------------------|----------------------|------------|-------------------|
|                        |            | $EM\uparrow$      | Tokens ↓   | $1/\rho\uparrow$  | $EM\uparrow$         | Tokens ↓   | $1/\rho \uparrow$ |
| GSM8K                  |            |                   |            |                   |                      |            |                   |
| Selective-Context [20] | EMNLP'2023 | 76.57             | 436        | 5.4x              | 76.15                | 182        | 13.0x             |
| LLMLingua[18]          | EMNLP'2023 | 76.72             | 462        | 5.1x              | 77.02                | 174        | 13.6x             |
| LLMLingua-2-small [12] | ACL'2024   | 75.66             | 425        | 5.6x              | 76.80                | <u>151</u> | 15.7x             |
| LLMLingua-2 [12]       | ACL'2024   | 76.87             | <u>415</u> | <u>5.7x</u>       | 76.80                | 140        | 16.9x             |
| LLM-DCP (Ours)         | -          | 77.03             | 343        | $\overline{6.9x}$ | 77.03                | 153        | 15.5x             |
| ВВН                    |            |                   |            |                   |                      |            |                   |
| Selective-Context [20] | EMNLP'2023 | 82.81             | 278        | 2.8x              | 81.91                | 152        | 5.1x              |
| LLMLingua[18]          | EMNLP'2023 | 81.68             | 271        | 2.9x              | 84.72                | 162        | 4.8x              |
| LLMLingua-2-small [12] | ACL'2024   | 82.73             | 274        | 2.8x              | 82.12                | 155        | 5.0x              |
| LLMLingua-2 [12]       | ACL'2024   | 82.41             | <u>255</u> | 3.0x              | 82.64                | 145        | 5.3x              |
| LLM-DCP (Ours)         | -          | 83.16             | <b>251</b> | $\overline{3.1x}$ | 83.98                | 145        | 5.3x              |

TABLE II
PERFORMANCE OF DIFFERENT METHODS ON THE REASONING (GSM8K), AND IN-CONTEXT LEARNING (BBH) TASKS.

- LLMLingua-2 [12]: It treats prompt compression as a token classification task, and it is available in LLMLingua-2-small and LLMLingua-2 versions.
- 2) Datasets: To comprehensively evaluate the effectiveness of the proposed LLM-DCP, we conduct experiments on four different datasets on summarization, conversation, reasoning, and In-context learning (ICL) tasks.
  - Arxiv-March23: It is a dataset consisting of the latest academic papers from the arXiv preprint repository, collected since March 2023. For our experimental evaluation, we employ a subset of 500 entries sourced from the dataset created by Li et al. [20], which includes only the first two sections of each article to avoid excessive length.
  - ShareGPT: A dataset of 90k conversations collected from sharegpt.com<sup>1</sup>, involving multiple rounds of dialogue between users and ChatGPT in multiple languages and scenarios. We test the conversation task using sharegpt575 [20], which contains 575 multi-round dialogue examples.
  - GSM8K [59]: A widely used dataset for testing logic and mathematics in language modeling, containing 8.5k highquality linguistically diverse mathematical problems.
  - BBH [60]: A subset of the BIG-Bench dataset [61], it focuses on a set of 23 challenging tasks covering multi-step arithmetic, algorithmic reasoning, language comprehension, and world knowledge. It is specifically designed to assess CoT prompting. For our experiments, we chose six tasks to test, including *Boolean Expressions*, Causal Judgement, Date Understanding, Disambiguation QA, Dyck Languages, and Formal Fallacies.
- 3) Evaluation Metrics: Following the settings of LLMLingua [18], we use BLEU [62], BLEURT [63], ROUGE [64] and BERTScore (BS F1) [54] as evaluation metrics for Arxiv-March and ShareGPT. We use Exact Match (EM) [18] as a metric for GSM8K and BBH. In addition, the compression ratio  $(1/\rho)$  is also included in the assessment metrics to ensure fairness. Note that we use the tokenizer of Llama $3^2$  to calculate the number of tokens.

4) Implementation Details: Our proposed LLM-DCP is implemented using PyTorch framework with Pytorch version 2.1.2 and runs on the 80G memory-sized NVIDIA A800 GPU with CUDA version 12.1. We use Adam as our optimizer to update the parameters of neural networks. The learning rate is set to  $10^{-5}$  for the actor model and  $10^{-6}$  for the critic model. The batch size is set to 4 and a total of 4 epochs are trained. The first and second stages are trained for 1 epoch respectively, and the third stage is trained for 2 epochs. The  $P_s$  and  $P_l$  in Eq. 4 are set to 200 and 100, respectively.

For the training of model  $M_s$  in subsection IV-D, we use the alpaca-gpt4-data<sup>3</sup> dataset (randomly selected 80% for the training set and 20% for the test set) to fine-tune Llama3-8B, and the training framework used is LLaMA-Factory<sup>4</sup>. Notably, the training hyperparameters use the default settings for full fine-tuning provided by LLaMA-Factory. We randomly selected 2048 prompt samples from the alpaca-gpt4-data dataset as training data to train the DCP-Agent. During the testing phase, we control the compression rate (e.g. 3x or 10x) by controlling the maximum step size. We employ the GPT-4o-mini-2024-07-18<sup>5</sup> as the target LLMs, with greedy decoding at a temperature of 0 for enhanced stability across experiments.

#### <span id="page-6-0"></span>B. Comparison Experiments

We compare the proposed LLM-DCP and three SOTA prompt compression methods to demonstrate the superior performance of our proposed method. We conduct experiments on a variety of downstream tasks, including conversation task (see Table I), summarization task (see Table II), reasoning task (see Table II), and In-context learning task (see Table II).

Excellent performance of the LLM-DCP in the conversation task. As shown in Table I, LLM-DCP outperforms other SOTA methods in the conversation task. Specifically, compared to LLMLingua-2, the proposed LLM-DCP improves about 4.8% ( $61.97 \rightarrow 64.93$ ) on BLEU and about 1.0% ( $90.87 \rightarrow 91.80$ ) on BS F1 at higher compression ratio (3.3x

<span id="page-6-1"></span><sup>1</sup>https://sharegpt.com/

<span id="page-6-2"></span><sup>&</sup>lt;sup>2</sup>https://huggingface.co/meta-llama/Meta-Llama-3-8B

<span id="page-6-3"></span><sup>3</sup>https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data

<span id="page-6-4"></span><sup>&</sup>lt;sup>4</sup>https://github.com/hiyouga/LLaMA-Factory

<span id="page-6-5"></span><sup>&</sup>lt;sup>5</sup>https://platform.openai.com/

