# B. Proof of Theorem 2

*Proof.* First, taking the derivative of  $g(\tilde{y}_l)$ , we can obtain:  $g'(\tilde{y}_l) = (c^c\theta_1 - c^c\theta_1\theta_2\tilde{y}_l - H^w\theta_1\theta_2)\exp(-\theta_2\tilde{y}_l) + c^c(\theta_3 + \frac{t_l}{\tilde{s}_l})$  The second derivative is:

$$g''(\tilde{y}_l) = c^c \theta_1 \theta_2^2 \exp(-\theta_2 \tilde{y}_l) [\tilde{y}_l - (\frac{2}{\theta_2} - \frac{H^w}{c^c})]$$

Since  $c^c \theta_1 \theta_2^2 \exp(-\theta_2 \tilde{y}_l) > 0$ ,  $g''(\tilde{y}_l)$  is a monotonically increasing function. Its zero point is  $\tilde{y}_l = \frac{2}{\theta_2} - \frac{H^w}{c^c}$ . Therefore, when  $\tilde{y}_l \geqslant \frac{2}{\theta_2} - \frac{H^w}{c^c}$ ,  $g''(\tilde{y}_l) \geqslant 0$ , and the function is convex. Meanwhile, since  $\frac{2}{\theta_2}$  is convex on  $(0,\infty)$  and  $\frac{H^w}{c^c}$  is constant, the function  $\frac{2}{\theta_2} - \frac{H^w}{c^c}$  is also convex on this interval. Furthermore, since  $\frac{d}{d\theta_2} \left( \frac{2}{\theta_2} - \frac{H^w}{c^c} \right) = -\frac{2}{\theta_2^2} < 0$ , the term  $\frac{2}{\theta_2} - \frac{H^w}{c^c}$  is monotonically decreasing in its domain. Therefore, when  $\theta_2 \geqslant \frac{2c^c}{H^w}$ , it implies that  $\frac{2}{\theta_2} - \frac{H^w}{c^c} \leqslant 0 < \tilde{y}_l$ , ensuring  $g(\tilde{y}_l)$  is strictly convex on  $(0,\infty)$ .

## REFERENCES

- <span id="page-9-0"></span> F. Barreto, L. Moharkar, M. Shirodkar, V. Sarode, S. Gonsalves, and A. Johns, "Generative artificial intelligence: Opportunities and challenges of large language models," in *Proc. of ICICN*. Springer, 2023, pp. 545–553.
- <span id="page-9-1"></span>[2] X. Ma, G. Fang, and X. Wang, "Llm-pruner: On the structural pruning of large language models," Advances in neural information processing systems, vol. 36, pp. 21702–21720, 2023.
- <span id="page-9-2"></span>[3] B. Lin, Z. Tang, Y. Ye, J. Cui, B. Zhu, P. Jin, J. Huang, J. Zhang, Y. Pang, M. Ning et al., "Moe-llava: Mixture of experts for large vision-language models," arXiv preprint arXiv:2401.15947, 2024.
- <span id="page-9-3"></span>[4] J. Duan, S. Qian, D. Yang, H. Hu, J. Cao, and G. Xue, "Mopar: A model partitioning framework for deep learning inference services on serverless platforms," arXiv preprint arXiv:2404.02445, 2024.
- <span id="page-9-4"></span>[5] Y. Li, Y. Lin, Y. Wang, K. Ye, and C. Xu, "Serverless computing: state-of-the-art, challenges and opportunities," *IEEE Transactions on Services Computing*, vol. 16, no. 2, pp. 1522–1539, 2022.
- <span id="page-9-5"></span>[6] Y. Fu, L. Xue, Y. Huang, A.-O. Brabete, D. Ustiugov, Y. Patel, and L. Mai, "{ServerlessLLM}:{Low-Latency} serverless inference for large language models," in *Proc. of OSDI*, 2024, pp. 135–153.
- <span id="page-9-6"></span>[7] H. Yu, X. Cui, H. Zhang, and H. Wang, "fmoe: Fine-grained expert offloading for large mixture-of-experts serving," arXiv preprint arXiv:2502.05370, 2025.
- <span id="page-9-34"></span>[8] L. Xue, Y. Fu, Z. Lu, C. Sun, L. Mai, and M. K. Marina, "Moe-infinity: Efficient moe inference on personal machines with sparsity-aware expert cache," 2025.
- <span id="page-9-8"></span>[9] X. Song, Z. Zhong, R. Chen, and H. Chen, "Promoe: Fast moe-based llm serving using proactive caching," arXiv preprint arXiv:2410.22134, 2024.
- <span id="page-9-7"></span>[10] P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, and M. Guo, "Hobbit: A mixed precision expert offloading system for fast moe inference," arXiv preprint arXiv:2411.01433, 2024.
- <span id="page-9-9"></span>[11] V. Gupta, K. Sinha, A. Gavrilovska, and A. P. Iyer, "Lynx: Enabling efficient moe inference through dynamic batch-aware expert selection," arXiv preprint arXiv:2411.08982, 2024.
- <span id="page-9-10"></span>[12] M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmann et al., "Phi-4 technical report," arXiv preprint arXiv:2412.08905, 2024.
- <span id="page-9-11"></span>[13] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu et al., "Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models," arXiv preprint arXiv:2401.06066, 2024.
- <span id="page-9-12"></span>[14] M. Liu, W. Wang, and C. Wu, "Optimizing distributed deployment of mixture-of-experts model inference in serverless computing," in *Proc.* of INFOCOM. IEEE, 2025, pp. 1–10.
- <span id="page-9-13"></span>[15] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan et al., "Deepseek-v3 technical report," arXiv preprint arXiv:2412.19437, 2024.
- <span id="page-9-14"></span>[16] R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, and M. Yang, "Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference," in *Proc. of ISCA*. IEEE, 2024, pp. 1018– 1031.
- <span id="page-9-15"></span>[17] KVCache-AI, "Ktransformers: A flexible framework for experiencing cutting-edge llm inference optimizations," https://github.com/kvcache-ai/ktransformers, 2024.
- <span id="page-9-16"></span>[18] S. Zhong, Y. Sun, L. Liang, R. Wang, R. Huang, and M. Li, "Hybrimoe: Hybrid cpu-gpu scheduling and cache management for efficient moe inference," arXiv preprint arXiv:2504.05897, 2025.

- <span id="page-9-17"></span>[19] M. Yu, Z. Jiang, H. C. Ng, W. Wang, R. Chen, and B. Li, "Gillis: Serving large neural networks in serverless functions with automatic model partitioning," in *Proc. of ICDCS*. IEEE, 2021, pp. 138–148.
- <span id="page-9-18"></span>[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," Advances in neural information processing systems, vol. 30, 2017.
- <span id="page-9-19"></span>neural information processing systems, vol. 30, 2017.

  [21] C. Helmberg and F. Rendl, "Solving quadratic (0, 1)-problems by semidefinite programs and cutting planes," *Mathematical programming*, vol. 82, no. 3, pp. 291–315, 1998.
- <span id="page-9-20"></span>[22] L. Zheng, W.-L. Chiang, Y. Sheng, T. Li, S. Zhuang, Z. Wu, Y. Zhuang, Z. Li, Z. Lin, E. P. Xing et al., "Lmsys-chat-1m: A large-scale real-world llm conversation dataset," arXiv preprint arXiv:2309.11998, 2023.
- <span id="page-9-21"></span>[23] L. Lee, "On the effectiveness of the skew divergence for statistical language analysis," in *International workshop on artificial intelligence* and statistics. PMLR, 2001, pp. 176–183.
- <span id="page-9-22"></span>[24] P. Sitikhu, K. Pahi, P. Thapa, and S. Shakya, "A comparison of semantic similarity methods for maximum human interpretability," in *Proc. of AITB*, vol. 1. IEEE, 2019, pp. 1–4.
- <span id="page-9-23"></span>[25] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, A. Shrivastava, C. Zhang, Y. Tian, C. Re et al., "Deja vu: Contextual sparsity for efficient llms at inference time," in *Proc. of ICML*. PMLR, 2023, pp. 22137– 22176.
- <span id="page-9-24"></span>[26] R. L. Graham, "Bounds for certain multiprocessing anomalies," *Bell system technical journal*, vol. 45, no. 9, pp. 1563–1581, 1966.
- <span id="page-9-25"></span>[27] S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," arXiv preprint arXiv:1609.07843, 2016.
- <span id="page-9-26"></span>[28] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer," *Journal of machine learning* research, vol. 21, no. 140, pp. 1–67, 2020.
- <span id="page-9-27"></span>[29] Hugging Face, https://huggingface.co/datasets/cerebras/ SlimPajama-627B, June 2023.
- <span id="page-9-28"></span>[30] L. Kaufman and P. J. Rousseeuw, Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
- <span id="page-9-29"></span>[31] H. Ma, Z. Du, and Y. Chen, "Moe-gps: Guidlines for prediction strategy for dynamic expert duplication in moe load balancing," arXiv preprint arXiv:2506.07366, 2025.
- <span id="page-9-30"></span>[32] Z. Fang, Z. Hong, Y. Huang, Y. Lyu, W. Chen, Y. Yu, F. Yu, and Z. Zheng, "Accurate expert predictions in moe inference via cross-layer gate," arXiv e-prints, pp. arXiv-2502, 2025.
- <span id="page-9-31"></span>[33] C. Lou, S. Qi, C. Jin, D. Nie, H. Yang, X. Liu, and X. Jin, "To-wards swift serverless llm cold starts with paraserve," arXiv preprint arXiv:2502.15524, 2025.
- <span id="page-9-32"></span>[34] C. Xu, Z. Li, Q. Chen, H. Zhao, and M. Guo, "Llm-mesh: Enabling elastic sharing for serverless llm inference," arXiv preprint arXiv:2507.00507, 2025.
- <span id="page-9-33"></span>[35] T. Huang, P. Chen, K. Gong, J. Hawk, Z. Bright, W. Xie, K. Huang, and Z. Ji, "Enova: Autoscaling towards cost-effective and stable serverless llm serving," arXiv preprint arXiv:2407.09486, 2024.
- <span id="page-9-35"></span>[36] Y. Zhang, S. Aggarwal, and T. Mitra, "Daop: Data-aware offloading and predictive pre-calculation for efficient moe inference," in 2025 Design, Automation & Test in Europe Conference (DATE). IEEE, 2025, pp. 1–7
- <span id="page-9-36"></span>[37] X. He, S. Zhang, Y. Wang, H. Yin, Z. Zeng, S. Shi, Z. Tang, X. Chu, I. Tsang, and O. Y. Soon, "Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference," arXiv preprint arXiv:2410.17954, 2024.
- <span id="page-9-37"></span>[38] Anonymous, "Technical report," 2025, the technical report will be made publicly available upon acceptance of the manuscript due to the submission policy.