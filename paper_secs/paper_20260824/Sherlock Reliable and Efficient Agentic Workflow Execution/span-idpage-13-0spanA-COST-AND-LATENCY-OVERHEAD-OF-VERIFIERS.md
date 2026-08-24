# <span id="page-13-0"></span>A COST AND LATENCY OVERHEAD OF VERIFIERS

## A.1 Microbenchmarks

In this paper, we use following microbenchmarks.

- Instruction-following tasks: HotpotQA [\(Yang et al.,](#page-12-0) [2018\)](#page-12-0), DROP [\(Dua et al.,](#page-10-0) [2019\)](#page-10-0), Instruction [\(HuggingFaceH4,](#page-11-0) [2025a\)](#page-11-0)
- Math tasks: MATH [\(HuggingFaceH4,](#page-11-0) [2025b\)](#page-11-0), GSM8k [\(Cobbe et al.,](#page-10-0) [2021\)](#page-10-0)
- Coding tasks: Humaneval [\(Chen et al.,](#page-10-0) [2021\)](#page-10-0), MBPP [\(Austin et al.,](#page-10-0) [2021\)](#page-10-0)
- Tool-calling tasks: GTA [\(Wang et al.,](#page-12-0) [2024a\)](#page-12-0)

#### A.2 Verifier Cost Calculation

Each verifier incurs different costs, which can be calculated as the sum of model cost and GPU cost. Model cost refers to the expense of using closed-source APIs—for example, commercial LLMs—whereas open-source models like the LLaMA family [\(Dubey et al.,](#page-10-0) [2024\)](#page-10-0) and Qwen family [\(Bai et al.,](#page-10-0) [2023a\)](#page-10-0) do not incur model costs. However, serving these open-source models requires compute resources, leading to GPU costs.

Model cost typically depends on the number of tokens in the prompt (input) and response (output), with output tokens generally priced higher. GPU cost is determined by the number of GPUs used and the duration of use. Given unit cost, GPU cost can be calculate as follows.

$$GPU\_cost = \frac{unit\_price \times num\_gpus \times num\_tokens}{throughput_{max} \times cluster\_utilization_{avg}}$$
(9)

In our setup, we deploy open-source models for the executors, advanced feedback model, and judge on an 8-GPU server costing \$13.60 per hour (*unit price*). We allocate 2 GPUs to the main executor model, 1 GPU to secondary executor model, 1 GPU to the judge model, and 4 GPUs to the advanced feedback model.

## A.3 Judge LLM and Majority Answer

For Self-consistency, the majority answer can be obtained either by generation (sc-gen), where a new response is produced from the set of executor outputs, or by selection (sc-select), where an existing executor response closest to the majority is chosen. sc-gen provides higher-quality outputs by smoothing superficial variations (e.g., wording, formatting) but incurs additional generation cost. By contrast, sc-select is more efficient, requiring only a single token to identify the nearest executor, but cannot reconcile divergent responses when no exact overlap exists.

The relative effectiveness of sc-gen and sc-select varies across tasks, largely due to the specialization of the Judge LLM. Since the Judge model is fine-tuned specifically for evaluation, such specialization can degrade its general instructionfollowing ability [\(Luo et al.,](#page-11-0) [2025b;](#page-11-0) [Wang et al.,](#page-12-0) [2024b\)](#page-12-0), often leading it to disregard formatting constraints given in prompts. This negatively impacts tasks requiring strict output structure, such as code generation or tool invocation. Consequently, sc-gen with a general-purpose model (e.g., LLaMA-8B) outperforms sc-gen with the Judge LLM on Tools and Code tasks. In contrast, for sc-select, the Judge LLM proves more effective, yielding superior solutions on Tools and Code tasks and achieving comparable accuracy on QA tasks (Tab 4).

|                       | QA    |        | Tools | Code      |       |  |  |
|-----------------------|-------|--------|-------|-----------|-------|--|--|
|                       | Drop  | Hotpot | GTA   | Humaneval | MBPP  |  |  |
| No-verify             | 59.0% | 74.5%  | 65.8% | 60.8%     | 44.5% |  |  |
| sc-gen (Judge LLM)    | 58.0% | 70.5%  | 39.7% | 65.2%     | 49.9% |  |  |
| sc-gen (llama3-8b)    | 55.0% | 68.0%  | 50.4% | 65.9%     | 52.0% |  |  |
| sc-select (Judge LLM) | 62.0% | 75.5%  | 71.4% | 76.2%     | 58.1% |  |  |
| sc-select (llama3-8b) | 63.0% | 76.0%  | 67.7% | 69.5%     | 57.4% |  |  |

Table 4. Generation-based vs. Selection-based Majority Answer (sc-gen vs. sc-select)

