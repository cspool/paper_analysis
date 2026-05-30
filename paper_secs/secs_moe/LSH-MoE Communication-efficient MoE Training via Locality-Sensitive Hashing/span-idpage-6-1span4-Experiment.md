# <span id="page-6-1"></span>4 Experiment

#### 4.1 Implementation

Our LSH-MoE comprises a data compression/restoration component and a communication component. We utilize PyTorch 1.11 for developing the LSH clustering and NCCL for implementing the communication. Additionally, our method is framework-independent and can be easily applied to other MoE training frameworks such as Hetu-MoE [21, 26], DeepSpeed-MoE [29], and Tutel [12].

#### <span id="page-6-4"></span>4.2 Benchmarks and Datasets

Our evaluations are conducted by scaling pre-trained models equipped with MoE architecture across various application domains. This includes models like RoBERTa-MoE, T5-MoE and GPT-MoE in natural language processing (NLP), as well as Swin-MoE in computer vision (CV). Among these models, RoBERTa-MoE and T5-MoE are evaluated on pre-training task, while GPT-MoE and Swin-MoE undergo fine-tuning evaluation based on their official open-sourced model checkpoints <sup>1 2</sup>. We also evaluated the zero-shot accuracy of the pre-trained T5-MoE. Model configurations are detailed in Table 1.

<span id="page-6-2"></span><sup>1</sup>https://github.com/facebookresearch/fairseq/tree/main/examples/moe\_lm

<span id="page-6-3"></span> $<sup>^2</sup> https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md$ 

The RoBERTa-MoE model is pre-trained with masked language modeling tasks on a combined dataset, which includes BooksCorpus (∼ 800M words) and English Wikipedia (∼ 2,500M words). This dataset is tokenized using a tokenizer with a vocabulary size of 50,257. To assess the impact of our MoE method in compressing all-to-all communication on large model training, the T5-MoE model, which is with about 10B parameters, is pre-trained on an industry dataset (∼ 500M words) using a span-masked language modeling task. In addition to pre-training tasks, we further evaluate our work on fine-tuning tasks. To be specific, we fine-tune two open-sourced models, including the language model GPT-MoE on the General Language Understanding Evaluation (GLUE) benchmark and the vision model Swin-MoE on the ImageNet classification benchmark.

