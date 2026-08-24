# 8 Limitations

Our research has certain limitations: (1) Due to hardware constraints, our tests were limited to 400K tokens, which does not represent the upper bound of FocusLLM's capabilities. Future work will explore the full performance potential of FocusLLM and investigate the use of quantization methods to reduce operational costs. (2) While FocusLLM demonstrates exceptional training efficiency, we have observed that training on larger datasets can significantly enhance its generalizability and performance. Therefore, increasing the training data size will be a focus of future research.

## References

- <span id="page-8-13"></span>Zhangir Azerbayev, Edward Ayers, and Bartosz Piotrowski. 2023. [Proofpile: A pre-training dataset](https://huggingface.co/datasets/hoskinson-center/proof-pile) [of mathematical texts.](https://huggingface.co/datasets/hoskinson-center/proof-pile)
- <span id="page-8-9"></span>Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. *arXiv preprint arXiv:2308.14508*.
- <span id="page-8-19"></span>Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. 2024. Unlimiformer: Long-range transformers with unlimited length input. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-8-1"></span>Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023a. Extending context window of large language models via positional interpolation. *arXiv preprint arXiv:2306.15595*.
- <span id="page-8-15"></span>Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2023b. Longlora: Efficient fine-tuning of long-context large language models. *arXiv preprint arXiv:2309.12307*.
- <span id="page-8-5"></span>Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. *arXiv preprint arXiv:2305.14788*.
- <span id="page-8-6"></span>Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and Furu Wei. 2023. In-context autoencoder for context compression in a large language model. *arXiv preprint arXiv:2307.06945*.
- <span id="page-8-4"></span>Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. 2023. Lm-infinite: Simple on-the-fly length generalization for large language models. *arXiv preprint arXiv:2308.16137*.
- <span id="page-8-18"></span>Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, and Xia Hu. 2024. Llm maybe longlm: Self-extend llm context window without tuning. *arXiv preprint arXiv:2401.01325*.

- <span id="page-8-16"></span>Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang. 2023. How long can context length of open-source llms truly promise? In *NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following*.
- <span id="page-8-7"></span>Amirkeivan Mohtashami and Martin Jaggi. 2024. Random-access infinite context length for transformers. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-8-20"></span>Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. 2024. Leave no context behind: Efficient infinite context transformers with infiniattention. *arXiv preprint arXiv:2404.07143*.
- <span id="page-8-2"></span>Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. Yarn: Efficient context window extension of large language models. *arXiv preprint arXiv:2309.00071*.
- <span id="page-8-12"></span>Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. 2019. Compressive transformers for long-range sequence modelling. *arXiv preprint arXiv:1911.05507*.
- <span id="page-8-10"></span>Together. 2023b. [Redpajama: An open source recipe to](https://github.com/togethercomputer/RedPajama-Data) [reproduce llama training dataset.](https://github.com/togethercomputer/RedPajama-Data)
- <span id="page-8-11"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.
- <span id="page-8-8"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
- <span id="page-8-14"></span>Lewis Tunstall, Leandr Von Werra, and Thomas Wolf. 2022. Natural language processing with transformers.
- <span id="page-8-17"></span>Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miłos. 2024. Focused transformer: Contrastive train- ´ ing for context scaling. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-8-0"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. *Advances in neural information processing systems*, 30.
- <span id="page-8-3"></span>Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language models with self-generated instructions. *arXiv preprint arXiv:2212.10560*.

<span id="page-9-9"></span>Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, and Maosong Sun. 2024. Infllm: Unveiling the intrinsic capacity of llms for understanding extremely long sequences with training-free memory. *arXiv preprint arXiv:2402.04617*.

<span id="page-9-1"></span>Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient streaming language models with attention sinks. *arXiv preprint arXiv:2309.17453*.

<span id="page-9-0"></span>Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. 2023. Effective long-context scaling of foundation models. *arXiv preprint arXiv:2309.16039*.

<span id="page-9-5"></span>Howard Yen, Tianyu Gao, and Danqi Chen. 2024. Longcontext language modeling with parallel context encoding. *arXiv preprint arXiv:2402.16617*.

<span id="page-9-2"></span>Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024a. Soaring from 4k to 400k: Extending llm's context with activation beacon. *arXiv preprint arXiv:2401.03462*.

<span id="page-9-3"></span>Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024b. ∞[bench: Extending long](https://arxiv.org/abs/2402.13718) [context evaluation beyond 100k tokens.](https://arxiv.org/abs/2402.13718) *Preprint*, arXiv:2402.13718.

## <span id="page-9-4"></span>A Efficiency of FocusLLM

The parallel decoding mechanism of FocusLLM effectively reduces the computational complexity of the standard architecture. Specifically, when dealing with very long sequences, the primary computational burden in the transformer architecture lies in the attention mechanism, which has a complexity of O(L 2 ), where L represents the total sequence length. By dividing the sequence into n chunks, the complexity within each chunk becomes O((L/n) 2 ). Therefore, when we process chunks in parallel, the time complexity can be reduced to O((L/n) 2 ). And the space complexity of n chunks becomes approximately O((L/n) <sup>2</sup> ∗ n) = O(L <sup>2</sup>/n). This means that compared to a standard transformer, FocusLLM can reduce the computational complexity to a fraction, 1/n or even more of the original theoretically, where n is the number of chunks into which the sequence is divided. In experiments, the longer the sequence length, the more apparent the improvement in efficiency.

## <span id="page-9-6"></span>B Details of Training Data

We randomly sampled 80K sequences from Red-Pajama as our training corpus. Table [5](#page-9-10) shows the

detailed distribution.

<span id="page-9-10"></span>

| Length  | 3K∼4K | 4K∼6K | 6K∼8K | Total |
|---------|-------|-------|-------|-------|
| Count   | 30K   | 16K   | 34K   | 80K   |
| Portion | 38%   | 20%   | 42%   | 100%  |

Table 5: Length distribution of training corpus.

## <span id="page-9-7"></span>C Experimental Details

We primarily conduct experiments on the LLaMA2- 7B-Chat model. The additional trainable parameters mentioned in Section [2](#page-1-0) amount to only 2B approximately.

Specifically, we conducted training on a Linux server equipped with 8×A100 GPUs, each with 40GB of memory. The training was carried out for 10,000 steps, equivalent to one epoch of the entire training dataset, using a batch size of 8 and a learning rate of 5e-5 with a linear scheduler. To conserve GPU memory, we employed deepspeed's zero2\_offload optimizing stage. The training process was completed in approximately 20 hours.

For hyper-parameters, during training, the chunk size was randomly selected from the set {64, 128, 256, 1024, 2048}. For the length of tokens injected into each chunk, we set a default of 512 tokens for inference. And we ensured this length did not exceed the chunk size in the training procedure. As a result, the length of injected tokens was min{512, chunk size}. For evaluations on the Longbench, we adopt a larger local context size of 3,500 tokens for FocusLLM, consistent with the official setting.

