# 7 Conclusion

To address the critical challenge of efficiency in long-context LLM training, we introduce two training paradigms: SeCO and its enhanced variant SpaCO. By partitioning the input sequence into smaller, manageable chunks and performing localized backpropagation for each chunk, SeCO achieves substantial memory savings. Building upon this foundation, SpaCO introduces a carefully designed sparsification mechanism that randomly selects few chunks for backpropagation, reducing computational overhead. The integration of a mathematically-grounded compensation factor ensures unbiased gradient estimation. Our methods achieve impressive memory efficiency, enabling the fine-tuning of 8B models with 16K tokens on a single RTX 3090 GPU. This represents a 16× memory reduction compared to naive parallel training. SeCO and SpaCO significantly lower the barrier for practitioners working with long-context LLMs.

<span id="page-7-0"></span><sup>3</sup>Even in the absence of the compensation factor, excessively long gradient chains often result in vanishing or exploding gradients, diminishing their overall impact.

## Limitations

SeCO and SpaCO each present unique advantages but also have exhibit distinct limitations. SeCO achieves accurate gradient computation and efficient memory usage but suffers from a quadratic increase in computation with sequence length, making it impractical for training on ultra-long sequences. In contrast, SpaCO significantly reduces computational cost and maintains comparable memory efficiency but sacrifices gradient accuracy, introducing substantial randomness that complicates convergence. Ultimately, no single training strategy perfectly balances the trade-offs in all training scenarios. A practical approach requires identifying an optimal balance among the "impossible triangle" of computation, memory efficiency, and gradient accuracy.

### Ethics Statement

By optimizing memory consumption and computational efficiency, our approach not only lowers the financial barriers to training such models but also reduces energy consumption, contributing to more sustainable AI practices.

However, as with any significant technological advancement, ethical concerns must be considered. Lowering the cost and resource requirements for training long-context models may inadvertently enable the misuse of these models, including the creation of harmful or malicious language systems. It is essential to address these risks through responsible research practices and the development of robust safeguards.

### References

- <span id="page-8-1"></span>Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. 2024. [Training-free long-context scaling of large language](https://arxiv.org/abs/2402.17463) [models.](https://arxiv.org/abs/2402.17463) In *ICML*.
- <span id="page-8-17"></span>Léon Bottou and Olivier Bousquet. 2007. [The tradeoffs](https://papers.nips.cc/paper_files/paper/2007/hash/0d3180d672e08b4c5312dcdafdf6ef36-Abstract.html) [of large scale learning.](https://papers.nips.cc/paper_files/paper/2007/hash/0d3180d672e08b4c5312dcdafdf6ef36-Abstract.html) In *NeurIPS*.
- <span id="page-8-8"></span>Yaroslav Bulatov. 2018. [Fitting larger networks into](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) [memory.](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) Medium.
- <span id="page-8-9"></span>Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. [Training deep nets with sublinear](https://arxiv.org/abs/1604.06174) [memory cost.](https://arxiv.org/abs/1604.06174) *arXiv*.
- <span id="page-8-0"></span>Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2024. [Lon](https://arxiv.org/abs/2309.12307)[gloRA: Efficient fine-tuning of long-context large](https://arxiv.org/abs/2309.12307) [language models.](https://arxiv.org/abs/2309.12307) In *ICLR*.

- <span id="page-8-15"></span>Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga. 2016. [py](https://github.com/pytorch/pytorch/tree/v2.6.0)[torch/pytorch.](https://github.com/pytorch/pytorch/tree/v2.6.0) Github.
- <span id="page-8-12"></span>Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, and Ruslan Salakhutdinov. 2019. [Transformer-xl: Attentive language models](https://arxiv.org/abs/1901.02860) [beyond a fixed-length context.](https://arxiv.org/abs/1901.02860) In *ACL*.
- <span id="page-8-5"></span>Tri Dao. 2024. [Flashattention-2: Faster attention with](https://github.com/Dao-AILab/flash-attention) [better parallelism and work partitioning.](https://github.com/Dao-AILab/flash-attention) In *ICLR*.
- <span id="page-8-4"></span>Harm de Vries. 2023. [In the long \(context\) run.](https://www.harmdevries.com/post/context-length/) Personal website.
- <span id="page-8-20"></span>DeepSpeed. 2021. [Deepspeed's flops profiler.](https://www.deepspeed.ai/tutorials/flops-profiler/#flops-measurement) Deepspeed documentation.
- <span id="page-8-16"></span>Google. 2015. [tensorflow/tensorflow.](https://github.com/tensorflow/tensorflow/tree/v2.18.0) Github.
- <span id="page-8-11"></span>Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. 2022. [huggingface/ac](https://github.com/huggingface/accelerate/tree/v1.2.1)[celerate.](https://github.com/huggingface/accelerate/tree/v1.2.1) Github.
- <span id="page-8-7"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. [LoRA: Low-rank adaptation of](https://arxiv.org/abs/2106.09685) [large language models.](https://arxiv.org/abs/2106.09685) In *ICLR*.
- <span id="page-8-13"></span>Zhiyuan Hu, Yuliang Liu, Jinman Zhao, and other. 2024. [Longrecipe: Recipe for efficient long context gener](https://arxiv.org/abs/2409.00509)[alization in large language models.](https://arxiv.org/abs/2409.00509) *arXiv*.
- <span id="page-8-21"></span>Jared Kaplan. 2019. [Notes on contemporary machine](https://www.semanticscholar.org/paper/Notes-on-Contemporary-Machine-Learning-for-Kaplan/70a1e83b5c539eacfa972710c92ac4b6ac8d128d) [learning for physicists.](https://www.semanticscholar.org/paper/Notes-on-Contemporary-Machine-Learning-for-Kaplan/70a1e83b5c539eacfa972710c92ac4b6ac8d128d) Semantic Scholar.
- <span id="page-8-18"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. [Effi](https://arxiv.org/abs/2309.06180)[cient memory management for large language model](https://arxiv.org/abs/2309.06180) [serving with pagedattention.](https://arxiv.org/abs/2309.06180) In *SIGOPS*.
- <span id="page-8-3"></span>Jerry Liu. 2022. [run-llama/llama\\_index.](https://github.com/run-llama/llama_index/tree/v0.12.16) Github.
- <span id="page-8-6"></span>Meta-AI. 2024. [The llama 3 herd of models.](https://arxiv.org/abs/2407.21783) Technical report.
- <span id="page-8-10"></span>MicroSoft. 2021. [microsoft/deepspeed.](https://github.com/deepspeedai/DeepSpeed/tree/v0.16.3) Github.
- <span id="page-8-2"></span>Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. [YaRN: Efficient context window](https://arxiv.org/abs/2309.00071) [extension of large language models.](https://arxiv.org/abs/2309.00071) In *ICLR*.
- <span id="page-8-22"></span>Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, Chloe Hillier, and Timothy P Lillicrap. 2018. [google](https://github.com/google-deepmind/pg19)[deepmind/pg19.](https://github.com/google-deepmind/pg19) Github.
- <span id="page-8-14"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. [Attention is all](https://arxiv.org/abs/1706.03762) [you need.](https://arxiv.org/abs/1706.03762) In *NeurIPS*.
- <span id="page-8-19"></span>Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. [Flashinfer: Efficient and cus](https://arxiv.org/abs/2501.01005)[tomizable attention engine for llm inference serving.](https://arxiv.org/abs/2501.01005) *arXiv*.

<span id="page-9-0"></span>Jinman Zhao, Xueyan Zhang, et al. 2024. [Large lan](https://openreview.net/forum?id=wLQ3I0F1oj)[guage model is not a \(multilingual\) compositional](https://openreview.net/forum?id=wLQ3I0F1oj) [relation reasoner.](https://openreview.net/forum?id=wLQ3I0F1oj) In *CoLM*.

<span id="page-9-3"></span>Table 1: Arguments for DeepSpeed ZeRO3 offload

| Argument                    | Value |
|-----------------------------|-------|
| overlap_comm                | true  |
| contiguous_gradients        | true  |
| reduce_bucket_size          | 5e8   |
| stage3_max_live_parameters  | 1e9   |
| stage3_max_reuse_distance   | 1e9   |
| stage3_prefetch_bucket_size | 5e8   |

### A Experimental Datasets

PG19 Dataset. The PG19 corpus, an open-source long-text dataset released by DeepMind, is derived from books in the [Project Gutenberg](https://www.gutenberg.org) repository published prior to 1919. This collection is supplemented with metadata containing book titles and publication dates. For model training, we randomly selected 1,000 samples from the PG19 training partition. To ensure consistent sequence lengths, text samples exceeding 16K tokens were truncated to this threshold. The PG19 dataset is publicly available under the Apache License 2.0.

### B Language Models

LLaMA3-8B. The LLaMA3-8B model, an opensource large language model developed by Meta AI, serves as the foundational model in our experiments. This selection is motivated by its widespread adoption within the research community. The licensing terms for the LLaMA3 series models are governed by the [Meta Llama 3 Com](https://github.com/meta-llama/llama3/blob/main/LICENSE)[munity License Agreement,](https://github.com/meta-llama/llama3/blob/main/LICENSE) which notably permits academic and commercial use with specific attribution requirements.

### C Implementation Details

### C.1 Pseudocode

The workflows of SeCO and SpaCO primarily manage the KV cache, focusing on its updates and the relay of gradients during backpropagation. These operations require overriding the default backpropagation mechanism in deep learning frameworks, which poses implementation challenges. To clarify this process, we provide pseudocode below.

<span id="page-9-4"></span>Table 2: Training results of SeCO vs. Model Parallelism (Baseline) across different learning rates.

| Method   | LR   |      |      |
|----------|------|------|------|
|          | 1e-4 | 3e-4 | 1e-3 |
| Baseline | 2.52 | 2.16 | 2.13 |
| SeCO     | 2.53 | 2.18 | 2.15 |

### <span id="page-9-2"></span>C.2 ZeRO3 Offload

Detailed configurations are provided in Table [1.](#page-9-3)

### <span id="page-9-1"></span>D Additional Results

Direct Validation of Gradient Accuracy. To assess the accuracy of the computed gradients, we conducted experiments using Qwen2.5-0.5B with float64 precision. Gradients were obtained for sequences of 512 tokens using both naive parallel training and SeCO (with a chunk size of 64) and then compared element-wise. The results show that the gradients computed with SeCO achieve a precision exceeding 12 decimal places. The test code for this experiment is publicly available in our repository under the test\_estimate directory.

Indirect Validation of Gradient Accuracy. To evaluate SeCO's performance in real training scenarios, we follow the experimental setup described in the main text. We compare SeCO's training results with those obtained using model parallelism and gradient checkpointing. The results are summarized in Table [2.](#page-9-4)

The minor performance gap may be attributed to numerical issues arising from the increased number of operations in SeCO. For example, FlashAttention introduces randomness during backpropagation due to the use of atomic additions (see [Github](https://github.com/Dao-AILab/flash-attention/issues/414) [issue\)](https://github.com/Dao-AILab/flash-attention/issues/414). Since SeCO involves tens of times more such operations than parallel training, it exhibits greater numerical instability.

```
1 def update_kv_cache(kv_cache, keys, vals):
2 try:
3 return concat(kv_cache.keys, keys), concat(kv_cache.vals,
               vals)
4 finally:
5 if is_gradient_enabled():
6 kv_cache.keys.append(keys)
7 kv_cache.vals.append(vals)
8 else:
9 k_detach, v_detach = keys.detach(), vals.detach()
10 k_detach.requires_grad_(), v_detach.requires_grad_()
11 kv_cache.keys.append(k_detach)
12 kv_cache.vals.append(v_detach)
13
14 def grad_hook(grad, base, scaler=1):
15 return grad + base * scaler
16
17 def copy_grad(a, b):
18 for ak, av, bk, bv in zip(a.keys, a.vals, b.keys, b.vals):
19 bk.register_hook(partial(grad_hook, base=ak.grad))
20 bv.register_hook(partial(grad_hook, base=av.grad))
```