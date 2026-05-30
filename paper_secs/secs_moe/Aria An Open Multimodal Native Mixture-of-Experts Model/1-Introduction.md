# **1 Introduction**

In this report, we present ARIA, the first open mixture-of-experts (MoE) model that is **multimodal native**. The term multimodal native has been used in prior literature to refer to different model capabilities, without a clear consensus. Here, we provide a quantifiable definition: *A multimodal native model refers to a single model with strong understanding capabilities across multiple input modalities (e.g. text, code, image, video), that matches or exceeds the modalityspecialized models of similar capacities.* Our definition aligns with the user experience of proprietary multimodal models such as GPT-4o or Gemini-1.5, where a user does not need to differentiate inputs from different modalities. Instead, the model is expected to seamlessly handle and integrate multiple modalities' input with a single model.

While proprietary multimodal native models are not uncommon, their training recipes remain largely undisclosed. As a result, most open-source models are modal-specialized or show subpar performance across modalities. In this research, we fill the gap and introduce training recipes for developing multimodal native models from scratch, which includes key aspects below:

• **Model Architecture.** The core of our model is a fine-grained mixture-of-experts decoder, which enables faster training and inference speed over dense decoders, due to more efficient parameter utilization through expert specialization. ARIA MoE activates 3.5B parameters per text token and has a total of 24.9B parameters. Visual input of variable length, size, and aspect is encoded as visual tokens using a

| Category          | Benchmark             | ARIA | Pixtral-12B | Llama3.2-11B | GPT-4V | mini<br>GPT-4o | GPT-4o | Flash<br>Gemini-1.5 | Pro<br>Gemini-1.5 |
|-------------------|-----------------------|------|-------------|--------------|--------|----------------|--------|---------------------|-------------------|
| Knowledge/Math    | MMMU (val)            | 54.9 | 52.5        | 50.7         | 56.4   | 59.4           | 69.1   | 56.1                | 62.2              |
| (Multimodal)      | MathVista (testmini)  | 66.1 | 58.0        | 51.5         | -      | 54.7           | 63.8   | 58.4                | 63.9              |
| Document/Chart/   | DocVQA (test)         | 92.6 | 90.7        | 88.4         | 88.4   | -              | 92.8   | 89.9                | 93.1              |
| Scene Text        | ChartQA (test)        | 86.4 | 81.8        | 83.4         | 78.4   | -              | 85.7   | 85.4                | 87.2              |
| Understanding     | TextVQA (val)         | 81.1 | -           | -            | 78.0   | -              | -      | 78.7                | 78.7              |
| General Visual QA | MMBench-1.1           | 80.3 | -           | -            | 79.8   | 76.0           | 82.2   | -                   | 73.9              |
|                   | EgoSchema (test)      | 66.8 | -           | -            | -      | -              | 72.2   | 65.7                | 72.2              |
| Long Video        | LongVideoBench (test) | 65.3 | 47.4        | 45.7         | 60.7   | 58.8           | 66.7   | 62.4                | 64.4              |
| Understanding     | VideoMME (w subs)     | 72.1 | 47.5        | 50.2         | 63.3   | 68.9           | 77.2   | 75.0                | 81.3              |
| Knowledge/Math/   | MMLU (5-shot)         | 73.3 | 69.2        | 69.4         | 86.4   | -              | 89.1   | 78.9                | 85.9              |
| Reasoning         | MATH (CoT)            | 50.8 | 48.1        | 51.9         | -      | 70.2           | 76.6   | -                   | -                 |
| (Language)        | ARC Challenge         | 91.0 | -           | 83.4         | -      | 96.4           | 96.7   | -                   | -                 |
| Coding            | HumanEval             | 73.2 | 72.0        | 72.6         | 67.0   | 87.2           | 90.2   | 74.3                | 84.1              |

<span id="page-1-0"></span>Table 1: Performance comparison across various multimodal and language benchmarks. Results of competing models are collected from verified official sources or reruned with official settings.

lightweight visual encoder of 438M parameters. ARIA has a long multimodal context window of 64k tokens.

- **Data.** ARIA is pre-trained on 6.4T language tokens and 400B multimodal tokens. We develop a rigorous procedure to curate high-quality data from a diverse set of sources. The multimodal pre-train data includes four major categories: interleaved image-text sequence from common crawl, synthetic image captions, documents transcriptions and question-answering pairs, synthetic video captions and questionanswering pairs.
- **Training Pipeline.** We design a 4-stage training pipeline, including language pre-training, multimodal pre-training, multimodal long-context pre-training, and multimodal post-training. Each stage is designed to progressively enhance certain model capabilities while maintaining those acquired in early stages. Our pipeline efficiently and effectively exploits the data and compute resources to maximize model performance.

Following this recipe, ARIA demonstrates state-of-the-art performance as an open multimodal native model. Compared to Pixtral-12B [\[Mixtral,](#page-23-0) [2024\]](#page-23-0) and Llama3.2-11B [\[Dubey](#page-22-0) [et al.,](#page-22-0) [2024\]](#page-22-0), Aria demonstrates superior performance across a wide range of multimodal, language, and coding tasks, while enjoying lower inference cost due to the fewer number of activated parameters. In addition, ARIA also performs on par with proprietary models such as GPT-4o and Gemini-1.5 on various multimodal tasks. The detailed benchmark results are present in Table [1.](#page-1-0)

We release ARIA under the Apache 2.0 license, free for both academic and commercial use. To facilitate easier adoption, we open-source a training framework that enables finetuning ARIA on a wide variety of data sources and formats, using as few as one GPU.

