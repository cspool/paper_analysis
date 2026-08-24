# <span id="page-5-0"></span>**4. Experimental Setting**

#### <span id="page-5-2"></span>**4.1. Tools**

In the training, we prepare a large and comprehensive tool set (Appendix [D\)](#page-14-2), but only sample a subset for each training instance to build diverse tool configurations ([§3.3\)](#page-4-1). We fix the following tool set in the evaluation for fair comparison.

- ∙ **Basic tools.** We use Tavily search API [1](#page-5-1) for web search, Python sandbox for Code interpreter, and build Faiss index with Qwen3-Embedding-8B [\[22\]](#page-11-3) for local search. Additionally, we also incorporate domain-specific functions, such as get\_flight\_status, to address specialized challenges within those domains.
- ∙ **Specialized LLMs.** We prompt GPT-5 [\[23\]](#page-11-4), GPT-5-mini [\[23\]](#page-11-4) as code writer, employ Qwen2.5-Coder-32B-Instruct [\[24\]](#page-11-5) as another code writer, and leverage Qwen2.5-Math-72B [\[25\]](#page-11-6), Qwen2.5-Math-7B [\[25\]](#page-11-6) as specialized math models.
- ∙ **Generalist LLMs.** We consider GPT-5, GPT-5-mini, Llama-3.3-70B-Instruct [\[26\]](#page-11-7), and Qwen3-32B [\[27\]](#page-11-8) as representative generalist models.

### **4.2. Baselines**

We compare Orchestrator-8B produced by ToolOrchestra to baseline orchestrators constructed by prompting LLMs. Additionally, we also compare to off-the-shelf monolithic LLM systems that are (1) not equipped with tools, (2) equipped with basic tools, and (3) using the expanded tool set that further includes specialized expert models and strong generalist models.

For off-the-shelf LLMs, we evaluate GPT-5, Claude Opus 4.1 [\[28\]](#page-11-9), Llama-3.3-70B-Instruct, Qwen3-235B-A22B [\[27\]](#page-11-8), Llama-3\_3-Nemotron-Super-49B-v1 [\[29\]](#page-11-10), Qwen3-8B [\[27\]](#page-11-8).

#### **4.3. Evaluation Configuration**

We conduct experiments on three popular benchmarks with complex reasoning: **Humanity's Last Exam (HLE)**, **FRAMES**, and 2 **-Bench**. Details about these three benchmarks are given in Appendix [B.](#page-14-3) Throughout the evaluation, we use the official price for proprietary models and leverage the pricing systems of

<span id="page-5-1"></span><sup>1</sup>[https://www](https://www.tavily.com/)*.*tavily*.*com/

<span id="page-6-2"></span>Table 1 | Comparison of Orchestrator-8B with baselines (prompt-based LLMs). Llama-Nemotron-49B denotes Llama-3.3-Nemotron-Super-49B-v1. Cost in US cents, Latency in minutes, are averaged between HLE and Frames. More efficiency statistics on 2 -Bench are in Table [16](#page-20-2) in Appendix. Basic tools include domain functions, search and code interpreter ([§4.1\)](#page-5-2). ↑ The higher the better. ↓ The lower the better. The results of existing SOTA are reported by [\[23\]](#page-11-4) † .

| Tools                               | Model(s)           | HLE (↑) | FRAMES (↑) | 2<br>𝜏<br>-Bench (↑) | Cost (↓) | Latency (↓) |
|-------------------------------------|--------------------|---------|------------|----------------------|----------|-------------|
| Existing                            | GPT-5              | 35.2    | –          | 84.2‡                | –        | –           |
| reported                            | o3                 | 24.3    | –          | 68.4                 | –        | –           |
| SOTA                                | GPT-4o             | 5.3     | –          | 43.8                 | –        | –           |
|                                     | Qwen3-8B           | 3.2     | 24.2       | –*                   | 0.2      | 0.6         |
|                                     | Llama-Nemotron-49B | 3.6     | 25.6       | –*                   | 0.4      | 1.1         |
|                                     | Llama-3.3-70B      | 3.8     | 32.4       | –*                   | 0.5      | 1.4         |
| No tool                             | Qwen3-235B-A22B    | 5.2     | 34.3       | –*                   | 2.6      | 3.3         |
|                                     | Claude Opus 4.1    | 11.7    | 58.2       | –*                   | 27.4     | 8.2         |
|                                     | GPT-5              | 23.4    | 66.3       | –*                   | 6.2      | 4.1         |
|                                     | Qwen3-8B           | 4.7     | 26.5       | 40.7                 | 1.3      | 2.2         |
|                                     | Llama-Nemotron-49B | 6.8     | 28.2       | 23.2                 | 2.5      | 3.5         |
|                                     | Llama-3.3-70B      | 4.6     | 42.3       | 17.6                 | 2.8      | 4.3         |
| Basic tools                         | Qwen3-235B-A22B    | 14.0    | 39.5       | 52.9                 | 12.3     | 10.2        |
|                                     | Claude Opus 4.1    | 19.8    | 63.5       | 46.0                 | 76.2     | 32.5        |
|                                     | GPT-5              | 35.1    | 74.0       | 77.7                 | 30.2     | 19.8        |
|                                     | Qwen3-8B           | 30.6    | 68.9       | 72.3                 | 27.6     | 18.3        |
|                                     | Llama-Nemotron-49B | 25.8    | 57.9       | 66.7                 | 25.6     | 17.1        |
| Basic tools,                        | Llama-3.3-70B      | 19.7    | 52.4       | 55.8                 | 19.7     | 13.4        |
| Specialized LLMs<br>Generalist LLMs | Qwen3-235B-A22B    | 32.8    | 74.2       | 75.6                 | 29.7     | 21.2        |
|                                     | Claude Opus 4.1    | 34.6    | 72.8       | 76.8                 | 52.5     | 25.6        |
|                                     | GPT-5              | 21.2    | 57.5       | 62.3                 | 17.8     | 13.6        |
|                                     | Orchestrator-8B    | 37.1    | 76.3       | 80.2                 | 9.2      | 8.2         |

<sup>†</sup> The HLE results of Existing reported SOTA are based on the full set, while other baselines and ours are only on the text-only subset. ‡ Due to implementation differences, we could not fully reproduce GPT-5's reported result (84.2) and only reached 77.7 in our experiments.

TogetherAI[2](#page-6-0) for open-source models. We set the inference temperature to 0 and allow maximum 50 turn for Orchestrator to solve a task.

### **4.4. Training Configuration**

We employ Qwen3-8B as the backbone LLM and train it on the GeneralThought-430K [3](#page-6-1) dataset in conjunction with synthetic data (S[3.3\)](#page-4-1). The training configuration uses a learning rate of 1e-6, a maximum input sequence length of 24,000, and a maximum generation length of 8,000, with a training batch size of 16 and a rollout batch size of 8. We allow maximum 50 turns for the Orchestrator to finish a task during rollout and use 16 NVIDIA H100 GPUs throughout the training.

