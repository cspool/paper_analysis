# *D. Installation*

The complete source code and environment setup instructions are hosted in our GitHub repository. We delegate the step-by-step installation guide to the repository's README.md.

## *E. Experiment workflow*

The experiment workflow is fully automated via provided shell scripts, which trigger inference across the five datasets under specified batch sizes.

#### *F. Evaluation and expected results*

Executing the workflow generates comprehensive execution logs. The primary evaluation metrics are the average Time Per Output Token (TPOT) and the GPU cache hit ratio.

Our scripts automatically parse these logs to extract both metrics, which are expected to demonstrate significant latency reductions consistent with Figures 12 and 13 (Setting S3) in the paper. For a complete reproducibility guide—including script execution, output parsing, and mapping terminal results to the paper's figures—please refer to the repository's README.md.

## *G. Experiment customization*

Users can modify the '–alpha' parameter in the execution script to adjust the expert substitution threshold. For the Setting S3 reproduction, α is default to 0.25.

#### REFERENCES

- [1] "Ktransformers: A flexible framework for experiencing cutting-edge llm inference optimizations." https://github. com/kvcache-ai/KTransformers, 2024.
- [2] "Llama.cpp: a c++ implementation enabling efficient llm inference on cpus." https://github.com/ggml-org/llama. cpp, 2024.
- [3] "Opencompass is an llm evaluation platform, supporting a wide range of models (llama3, mistral, internlm2,gpt-4,llama2, qwen,glm, claude, etc) over 100+ datasets." https://https://github.com/open-compass/ opencompass, 2024.
- [4] "This url describes some of the common llm inference metrics." https://docs.nvidia.com/nim/benchmarking/llm/ latest/metrics.html, 2024.
- [5] "This url introduces the llm deepseek-moe-16b-chat." https://huggingface.co/deepseek-ai/ deepseek-moe-16b-chat, 2024.
- [6] "This url introduces the llm qwen2-57ba14b-instruct." https://huggingface.co/Qwen/ Qwen2-57B-A14B-Instruct, 2024.
- [7] "This url introduces the llm /xverse-moe-a4.2b-chat." https://huggingface.co/xverse/XVERSE-MoE-A4. 2B-Chat, 2024.
- [8] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley *et al.*, "Deepspeed-inference: enabling efficient inference of transformer models at unprecedented scale," in *SC22: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2022, pp. 1–15.
- [9] R. Awasthi, S. P. Ramachandran, S. Mishra, D. Mahapatra, H. Arshad, A. Atreja, A. Bhattacharyya, A. Bhattad, N. Singh, J. B. Cywinski *et al.*, "Artificial intelligence in healthcare: 2024 year in review," *medRxiv*, pp. 2025–02, 2025.
- [10] Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi, "Piqa: Reasoning about physical commonsense in natural language," in *Thirty-Fourth AAAI Conference on Artificial Intelligence*, 2020.
- [11] W. Cai, J. Jiang, F. Wang, J. Tang, S. Kim, and J. Huang, "A survey on mixture of experts in large language models," *IEEE Transactions on Knowledge and Data Engineering*, 2025.
- [12] S. Cao, S. Liu, T. Griggs, P. Schafhalter, X. Liu, Y. Sheng, J. E. Gonzalez, M. Zaharia, and I. Stoica, "Moe-lightning: High-throughput moe inference on memory-constrained gpus," in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1*, 2025, pp. 715–730.
- [13] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan,

- S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. Mc-Candlish, I. Sutskever, and W. Zaremba, "Evaluating large language models trained on code," 2021.
- [14] T. Chen, S. Huang, Y. Xie, B. Jiao, D. Jiang, H. Zhou, J. Li, and F. Wei, "Task-specific expert pruning for sparse mixture-of-experts," *arXiv preprint arXiv:2206.00277*, 2022.
- [15] Z. Chkirbene, R. Hamila, A. Gouissem, and U. Devrim, "Large language models (llm) in industry: A survey of applications, challenges, and trends," in *2024 IEEE 21st International Conference on Smart Communities: Improving Quality of Life using AI, Robotics and IoT (HONET)*. IEEE, 2024, pp. 229–234.
- [16] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, "Think you have solved question answering? try arc, the ai2 reasoning challenge," *arXiv:1803.05457v1*, 2018.
- [17] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu *et al.*, "Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models," *arXiv preprint arXiv:2401.06066*, 2024.
- [18] U. Deshpande, T. Janssen, M. Srivatsa, and S. Sundararaman, "Moesaic: Shared mixture of experts," in *Proceedings of the 2024 ACM Symposium on Cloud Computing*, November 2024, pp. 434–442.
- [19] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022.
- [20] O. Friha, M. A. Ferrag, B. Kantarci, B. Cakmak, A. Ozgun, and N. Ghoualmi-Zine, "Llm-based edge intelligence: A comprehensive survey on architectures, applications, security and trustworthiness," *IEEE Open Journal of the Communications Society*, 2024.
- [21] X. He, S. Zhang, Y. Wang, H. Yin, Z. Zeng, S. Shi, Z. Tang, X. Chu, I. Tsang, and O. Y. Soon, "Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference," *arXiv preprint arXiv:2410.17954*, 2024.
- [22] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, "Measuring massive multitask language understanding," *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021.
- [23] X. Jiang, Y. Zhou, S. Cao, I. Stoica, and M. Yu, "Neo: Saving gpu memory crisis with cpu offloading for online llm inference," *arXiv preprint arXiv:2411.01142*, 2024.
- [24] M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer, "triviaqa: A Large Scale Distantly Supervised Challenge

- Dataset for Reading Comprehension," *arXiv e-prints*, p. arXiv:1705.03551, 2017.
- [25] H. Kang, Z. Yu, and C. Xiong, "Flame-moe: A transparent end-to-end research platform for mixture-of-experts language models," *arXiv preprint arXiv:2505.20225*, 2025.
- [26] H. Kim, N. Wang, Q. Xia, J. Huang, A. Yazdanbakhsh, and N. S. Kim, "Lia: A single-gpu llm inference acceleration with cooperative amx-enabled cpu-gpu computation and cxl offloading," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, June 2025, pp. 544–558.
- [27] R. Kong, Y. Li, W. Wang, L. Kong, and Y. Liu, "Serving moe models on resource-constrained edge devices via dynamic expert swapping," *IEEE Transactions on Computers*, 2025.
- [28] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy, "RACE: Large-scale ReAding comprehension dataset from examinations," in *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*. Copenhagen, Denmark: Association for Computational Linguistics, Sep. 2017, pp. 785–794. [Online]. Available: https://aclanthology.org/D17-1082
- [29] C. Li, Y. Yin, X. Wu, J. Zhu, Z. Gao, D. Niu, Q. Wu, X. Si, Y. Xie, C. Zhang *et al.*, "H2 llm: Hardware-dataflow co-exploration for heterogeneous hybrid-bonding-based low-batch llm inference," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, 2025, pp. 194–210.
- [30] N. Li, S. Guo, T. Zhang, M. Li, Z. Hong, Q. Zhou, X. Yuan, and H. Zhang, "The moe-empowered edge llms deployment: Architecture, challenges, and opportunities," *arXiv preprint arXiv:2502.08381*, 2025.
- [31] P. Li, Z. Zhang, P. Yadav, Y.-L. Sung, Y. Cheng, M. Bansal, and T. Chen, "Merge, then compress: Demystify efficient smoe with hints from its routing policy," *arXiv preprint arXiv:2310.01334*, 2023.
- [32] X. Lu, Q. Liu, Y. Xu, A. Zhou, S. Huang, B. Zhang, J. Yan, and H. Li, "Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models," *arXiv preprint arXiv:2402.14800*, 2024.
- [33] M. T. Pilehvar and J. Camacho-Collados, "Wic: the wordin-context dataset for evaluating context-sensitive meaning representations," *arXiv preprint arXiv:1808.09121*, 2018.
- [34] D. Rivkin, F. Hogan, A. Feriani, A. Konar, A. Sigal, X. Liu, and G. Dudek, "Aiot smart home via autonomous llm agents," *IEEE Internet of Things Journal*, 2024.
- [35] R. Sarkar, H. Liang, Z. Fan, Z. Wang, and C. Hao, "Edge-moe: Memory-efficient multi-task vision transformer architecture with task-level sparsity via mixtureof-experts," in *2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD)*. IEEE, October 2023, pp. 1–9.
- [36] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V.

- Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," *arXiv preprint arXiv:1701.06538*, 2017.
- [37] R. Singh and S. S. Gill, "Edge ai: a survey," *Internet of Things and Cyber-Physical Systems*, vol. 3, pp. 71–92, 2023.
- [38] X. Song, Z. Zhong, R. Chen, and H. Chen, "Promoe: Fast moe-based llm serving using proactive caching," *arXiv preprint arXiv:2410.22134*, 2024.
- [39] Y. Song, Z. Mi, H. Xie, and H. Chen, "Powerinfer: Fast large language model serving with a consumergrade gpu," in *Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles*, 2024, pp. 590–606.
- [40] B. Spector and C. Re, "Accelerating llm inference with staged speculative decoding," *arXiv preprint arXiv:2308.04623*, 2023.
- [41] H. Wang, Q. Zhou, Z. Hong, and S. Guo, "D2moe: Dual routing and dynamic scheduling for efficient on-device moe-based llm serving," in *Proceedings of the 31st Annual International Conference on Mobile Computing and Networking*, 2025, pp. 574–588.
- [42] W. Wang, L. Mi, S. Cen, H. Dai, Y. Li, X. Fu, and Y. Liu, "Region-based content enhancement for {Efficient} video analytics at the edge," in *22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25)*, 2025, pp. 613–633.
- [43] L. Xue, Y. Fu, Z. Lu, L. Mai, and M. Marina, "Moeinfinity: Activation-aware expert offloading for efficient moe serving," *arXiv e-prints*, pp. arXiv–2401, 2024.
- [44] Z. Xue, Y. Song, Z. Mi, X. Zheng, Y. Xia, and H. Chen, "Powerinfer-2: Fast large language model inference on a smartphone," *arXiv preprint arXiv:2406.06282*, 2024.
- [45] P. Yadav, L. Choshen, C. Raffel, and M. Bansal, "Compeft: Compression for communicating parameter efficient updates via sparsification and quantization," *arXiv preprint arXiv:2311.13171*, 2023.
- [46] Z. Yang, Y. Hu, S. Sun, and W. Ji, "Ec2moe: Adaptive end-cloud pipeline collaboration enabling scalable mixture-of-experts inference," *arXiv preprint arXiv:2508.06024*, 2025.
- [47] Z. Yu, S. Liang, T. Ma, Y. Cai, Z. Nan, D. Huang, T. Chen *et al.*, "Cambricon-llm: A chiplet-based hybrid architecture for on-device inference of 70b llm," in *2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)*. IEEE, November 2024, pp. 1474– 1488.
- [48] D. Zhang, H. Zheng, W. Yue, and X. Wang, "Advancing its applications with llms: A survey on traffic management, transportation safety, and autonomous driving," in *International Joint Conference on Rough Sets*. Springer, 2024, pp. 295–309.
- [49] M. Zhang, X. Shen, J. Cao, Z. Cui, and S. Jiang, "Edgeshard: Efficient llm inference via collaborative edge computing," *IEEE Internet of Things Journal*, 2024.
- [50] Y. Zhang, S. Aggarwal, and T. Mitra, "Daop: Data-aware

- offloading and predictive pre-calculation for efficient moe inference," in *2025 Design, Automation & Test in Europe Conference (DATE)*. IEEE, March 2025, pp. 1–7.
- [51] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, "Judging llm-as-a-judge with mt-bench and chatbot arena," 2023.
- [52] S. Zhong, Y. Sun, L. Liang, R. Wang, R. Huang, and M. Li, "Hybrimoe: Hybrid cpu-gpu scheduling and cache management for efficient moe inference," *arXiv preprint arXiv:2504.05897*, 2025.
- [53] T. Zhong, Z. Chi, L. Gu, Y. Wang, Y. Yu, and J. Tang, "Meta-dmoe: Adapting to domain shift by metadistillation from mixture-of-experts," *Advances in Neural Information Processing Systems*, vol. 35, pp. 22 243– 22 257, 2022.