# 7 Conclusion and Future Works

The pursuit of efficient autonomous agents requires a fundamental shift in perspective: from micro-optimizing isolated components to macro-optimizing the end-to-end task workflow. Our findings demonstrate that naïve acceleration can be a double-edged sword, often introducing overheads that negate potential gains. A successful strategy must be holistic and synergistic, intelligently orchestrating architectural optimizations like AgentCollab and AgentCompress with inference-level strategies like AgentSched and AgentSAM. By adopting this task-centric view, AgentInfer successfully reduces ineffective token consumption by over 50% and achieves 1.8x-2.5x speedup on end-to-end latency, proving that efficiency need not come at the cost of cognitive capability. These results pave the way for a future of truly fast, reliable, and scalable autonomous agent systems that are optimized not just for speed, but for the successful completion of complex, long-horizon tasks.

## Acknowledgments

We would like to thank our Project Members for their valuable contributions and support throughout this project:

Contributors: Jie Chen, Jiahong Zhang, Yijun Hong, Fang Guo

Project Sponsors: Mingming Zhu, Yaoyuan Wang, Zhenhua Dong, Peifeng Qin, Baochuan Yang, Yunhe Wang

## References

- <span id="page-19-0"></span>[1] H. Chen, J. Qin, J. Guo, T. Yuan, Y. Yin, H. Zhen, Y. Wang, J. Li, X. Meng, M. Zhang *et al.*, "Pangu light: Weight re-initialization for pruning and accelerating llms," *arXiv preprint arXiv:2505.20155*, 2025.
- <span id="page-19-1"></span>[2] Y. Huang, Y. Chen, H. Zhang, K. Li, H. Zhou, M. Fang, L. Yang, X. Li, L. Shang, S. Xu *et al.*, "Deep research agents: A systematic examination and roadmap," *arXiv preprint arXiv:2506.18096*, 2025.
- <span id="page-19-2"></span>[3] X. Li, Z. Xing, Y. Li, L. Qu, H.-L. Zhen, W. Liu, Y. Yao, S. J. Pan, and M. Yuan, "Kvtuner: Sensitivity-aware layer-wise mixed-precision kv cache quantization for efficient and nearly lossless llm inference," *arXiv preprint arXiv:2502.04420*, 2025.
- <span id="page-19-3"></span>[4] Z. Pei, L. Zou, H.-L. Zhen, X. Yu, W. Liu, S. J. Pan, M. Yuan, and B. Yu, "Cmoe: Fast carving of mixture-of-experts for efficient llm inference," *arXiv preprint arXiv:2502.04416*, 2025.
- <span id="page-19-4"></span>[5] Z. Pei, H.-L. Zhen, X. Yu, S. J. Pan, M. Yuan, and B. Yu, "Fusegpt: Learnable layers fusion of generative pre-trained transformers," *arXiv preprint arXiv:2411.14507*, 2024.
- <span id="page-19-5"></span>[6] S. Sun, Y. Li, X. Li, Y. Lian, W. Lin, H.-L. Zhen, Z. Yang, C. Chen, X. Yu, M. Yuan *et al.*, "Scaling up, speeding up: A benchmark of speculative decoding for efficient llm test-time scaling," *arXiv preprint arXiv:2509.04474*, 2025.
- <span id="page-19-6"></span>[7] X. Wu, K. Li, Y. Zhao, L. Zhang, L. Ou, H. Yin, Z. Zhang, Y. Jiang, P. Xie, F. Huang *et al.*, "Resum: Unlocking long-horizon search intelligence via context summarization," *arXiv preprint arXiv:2509.13313*, 2025.
- <span id="page-19-7"></span>[8] W. Lin, X. Li, Z. Yang, X. Fu, H.-L. Zhen, Y. Wang, X. Yu, W. Liu, X. Li, and M. Yuan, "Trimr: Verifier-based training-free thinking compression for efficient test-time scaling," *arXiv preprint arXiv:2505.17155*, 2025.
- <span id="page-19-8"></span>[9] A. Java, A. Khandelwal, S. Midigeshi, A. Halfaker, A. Deshpande, N. Goyal, A. Gupta, N. Natarajan, and A. Sharma, "Characterizing deep research: A benchmark and formal definition," *arXiv preprint arXiv:2508.04183*, 2025.
- <span id="page-19-9"></span>[10] Y. Hu, K. Wang, X. Zhang, F. Zhang, C. Li, H. Chen, and J. Zhang, "Sam decoding: Speculative decoding via suffix automaton," *ACL*, 2024.

- <span id="page-20-0"></span>[11] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer, "Qlora: Efficient finetuning of quantized large language models," *NeurIPS*, 2023.
- <span id="page-20-1"></span>[12] E. Frantar, S. Ashkboos, D. Alistarh, and T. Hoefler, "Gptq: Accurate post-training quantization for generative pretrained transformers," *arXiv preprint arXiv:2210.17323*, 2023.
- <span id="page-20-2"></span>[13] J. Lin, Z. Tang, S. Li, and S. Han, "Awq: Activation-aware weight quantization for llm compression and acceleration," *arXiv preprint arXiv:2306.00978*, 2024.
- <span id="page-20-3"></span>[14] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," *arXiv preprint arXiv:1503.02531*, 2015.
- <span id="page-20-4"></span>[15] T. Wang, W. Zhou, Y. Zeng, and X. Zhang, "Efficientvlm: Fast and accurate vision-language models via knowledge distillation and modal-adaptive pruning," in *Findings of the Association for Computational Linguistics: ACL 2023*, 2023, pp. 13 899–13 913.
- <span id="page-20-5"></span>[16] P. Lewis, E. Perez, A. Piktus, A. Fan *et al.*, "Retrieval-augmented generation for knowledge-intensive nlp tasks," *NeurIPS*, 2020.
- <span id="page-20-6"></span>[17] M. Kang, S. Lee, J. Baek, K. Kawaguchi, and S. J. Hwang, "Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks," *Advances in Neural Information Processing Systems*, vol. 36, pp. 48 573–48 602, 2023.
- <span id="page-20-7"></span>[18] W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang, "Memorybank: Enhancing large language models with long-term memory," vol. 38, no. 17, pp. 19 724–19 731, 2024.
- <span id="page-20-8"></span>[19] Y. Chen, Z. You, S. Zhang, H. Li, Y. Li, Y. Wang, and M. Tan, "Core context aware transformers for long context language modeling," *arXiv preprint arXiv:2412.12465*, 2024.
- <span id="page-20-9"></span>[20] J. S. Park, C. Cai, M. R. Chen, M. S. Bernstein, and P. L. Li, "Generative agents: Interactive simulacra of human behavior," *arXiv preprint arXiv:2304.03442*, 2023.
- <span id="page-20-10"></span>[21] A. Petrov, M. Sandler, A. Zhmoginov, N. Miller, and M. Vladymyrov, "Long context in-context compression by getting to the gist of gisting," *arXiv preprint arXiv:2504.08934*, 2025.
- <span id="page-20-11"></span>[22] Y. Wang and X. Chen, "Mirix: Multi-agent memory system for llm-based agents," *arXiv preprint arXiv:2507.07957*, 2025.
- <span id="page-20-12"></span>[23] S. Yao, D. Zhao, B. Yu, J. Cui, K. Narasimhan, and D. Radev, "Tree of thoughts: Deliberate problem solving with large language models," *arXiv preprint arXiv:2305.10601*, 2024.
- <span id="page-20-13"></span>[24] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language agents with verbal reinforcement learning," *Advances in neural information processing systems*, vol. 36, pp. 8634–8652, 2023.
- <span id="page-20-14"></span>[25] Y. Leviathan, M. Kalman, and J. Dean, "Fast inference from transformers via speculative decoding," *arXiv preprint arXiv:2302.01318*, 2023.
- <span id="page-20-15"></span>[26] S. Sun, Y. Li, X. Li, Y. Lian, W. Lin, H.-L. Zhen, Z. Yang, C. Chen, X. Yu, M. Yuan, and C. Ma, "Scaling up, speeding up: A benchmark of speculative decoding for efficient llm test-time scaling," *https://arxiv.org/abs/2509.04474*, 2025.
- <span id="page-20-16"></span>[27] Y. Tan, S. He, K. Liu, and J. Zhao, "The zero-step thinking: An empirical study of mode selection as harder early exit in reasoning models," *arXiv preprint arXiv:2510.19176*, 2025.
- <span id="page-20-17"></span>[28] C. Wu, B. Li, M. Gao, and Z. Wang, "From efficiency to adaptivity: A deeper look at adaptive reasoning in large language models," *arXiv preprint arXiv:2511.10788*, 2025.
- <span id="page-20-18"></span>[29] Z. Shi, S. Gao, L. Yan, Y. Feng, X. Chen, Z. Chen, D. Yin, S. Verberne, and Z. Ren, "Tool learning in the wild: Empowering language models as automatic tool agents," pp. 2222–2237, 2025.
- <span id="page-20-19"></span>[30] S. Wang, Z. Tan, Z. Chen, S. Zhou, T. Chen, and J. Li, "Anymac: Cascading flexible multi-agent collaboration via next-agent prediction," pp. 11 566–11 578, 2025.
- <span id="page-20-20"></span>[31] Y. Liu, Y. Cheng, J. Yao, Y. An, X. Chen, S. Feng, Y. Huang, S. Shen, R. Zhang, K. Du *et al.*, "Lmcache: An efficient kv cache layer for enterprise-scale llm inference," *arXiv preprint arXiv:2510.09665*, 2025.
- <span id="page-20-21"></span>[32] H. Yang, R. Zhang, M. Huang, W. Wang, Y. Tang, Y. Li, Y. Liu, and D. Zhang, "Kvshare: An llm service system with efficient and effective multi-tenant kv cache reuse," *arXiv preprint arXiv:2503.16525*, 2025.
- <span id="page-20-22"></span>[33] M. Zhang, H. Sun, J. Wang, S. Li, W. Ning, Q. Qi, Z. Zhuang, and J. Liao, "Clusterattn: Kv cache compression under intrinsic attention clustering," in *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2025, pp. 14 451–14 473.

- <span id="page-21-0"></span>[34] L. Haoyang, Y. Li, A. Tian, T. Tang, Z. Xu, X. Chen, H. Nicole, W. Dong, L. Qing, and L. Chen, "A survey on large language model acceleration based on kv cache management," *Transactions on Machine Learning Research*, 2025.
- <span id="page-21-1"></span>[35] W. Kwon, "vllm: An efficient inference engine for large language models," Ph.D. dissertation, UC Berkeley, 2025.
- <span id="page-21-2"></span>[36] L. Zheng, L. Yin, Z. Xie, C. L. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez *et al.*, "Sglang: Efficient execution of structured language model programs," *Advances in neural information processing systems*, vol. 37, pp. 62 557–62 583, 2024.
- <span id="page-21-3"></span>[37] openPangu Team, "openpangu-deepdiver-v2 technical report," Huawei, Tech. Rep., 2025, accessed: 2025-12-03. [Online]. Available: [https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver/blob/main/docs/](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver/blob/main/docs/openpangu-deepdiver-v2-tech-report.pdf) [openpangu-deepdiver-v2-tech-report.pdf](https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver/blob/main/docs/openpangu-deepdiver-v2-tech-report.pdf)
- <span id="page-21-4"></span>[38] P. Zhou, B. Leon, X. Ying, C. Zhang, Y. Shao, Q. Ye, D. Chong, Z. Jin, C. Xie, M. Cao *et al.*, "Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese," *arXiv preprint arXiv:2504.19314*, 2025.
- <span id="page-21-5"></span>[39] openPangu Team, "openpangu-embedded-7b," [https://ai.gitcode.com/ascend-tribe/](https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model) [openpangu-embedded-7b-model,](https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model) 2025, accessed: 2025-12-03.
- <span id="page-21-6"></span>[40] vLLM-Ascend Project, "vllm ascend plugin," [https://github.com/vllm-project/vllm-ascend,](https://github.com/vllm-project/vllm-ascend) 2025, accessed: 2025- 12-03.