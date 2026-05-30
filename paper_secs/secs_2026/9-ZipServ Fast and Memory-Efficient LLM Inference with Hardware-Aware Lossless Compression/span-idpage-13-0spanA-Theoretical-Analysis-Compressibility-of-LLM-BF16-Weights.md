# <span id="page-13-0"></span>A Theoretical Analysis: Compressibility of LLM BF16 Weights

We present the theoretical foundation showing why exponent distributions in LLM weights are highly skewed and exhibit top-K contiguity.

Following recent studies [13, 40, 63], we assume that weights  $w \in \mathbb{R}^D$  in a single layer (vectorized for analysis) follow a zero-mean normal distribution:

$$w \sim \mathcal{N}(0, \sigma^2 I)$$

A non-zero, normal BF16 number v is represented as  $v = (-1)^S \times 2^{E-127} \times (1.m_1...m_7)_2$ , where S is the sign bit, E is the 8-bit unsigned integer value of the exponent field, and  $(1.m_1...m_7)_2$  is the 7-bit mantissa with an implicit leading 1. The bias for the BF16 exponent is 127.

Let x = E - 127 be the actual exponent value. Any number using this specific exponent E will have a magnitude in the range  $[2^x, 2^{x+1})$ . Our analysis focuses on the probability distribution of this exponent value x (or equivalently, E), given that the weights w are drawn from  $\mathcal{N}(0, \sigma^2)$ . The redundancy arises if this distribution P(X = x) is highly skewed,

meaning some exponent values are far more common than others.

The probability of a single weight  $w_i$  falling into the magnitude range corresponding to a specific exponent x is:

$$P(X = x) = P(2^x \le |w_i| < 2^{x+1})$$

Note that this calculation is an approximation. We are calculating the probability of a value falling into the exponent's ideal magnitude range  $[2^x, 2^{x+1})$ , which simplifies the BF16 quantization process by ignoring rounding effects caused by the 7-bit mantissa. However, this serves as a robust approximation for analyzing the overall exponent distribution.

Given that  $w_i \sim \mathcal{N}(0, \sigma^2)$ , its Probability Density Function (PDF) is  $f(w_i) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-w_i^2/(2\sigma^2)}$ . The probability is the integral of this PDF over the positive and negative ranges:

$$P_{\sigma}(X=x) = 2 \times \int_{2^{x}}^{2^{x+1}} \frac{1}{\sqrt{2\pi\sigma^{2}}} e^{-t^{2}/(2\sigma^{2})} dt$$

This integral can be expressed using the error function (erf), defined as  $\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-u^2} du$ :

$$P_{\sigma}(X = x) = \operatorname{erf}\left(\frac{2^{x+1}}{\sigma\sqrt{2}}\right) - \operatorname{erf}\left(\frac{2^{x}}{\sigma\sqrt{2}}\right)$$

**Theorem A.1.** The function  $P(X = x) = erf\left(\frac{2^{x+1}}{\sigma\sqrt{2}}\right) - erf\left(\frac{2^x}{\sigma\sqrt{2}}\right)$  is unimodal for  $x \in \mathbb{Z}$ .

*Proof.* To prove unimodality, we consider the continuous extension  $f(x) = \operatorname{erf}\left(\frac{2^{x+1}}{\sigma\sqrt{2}}\right) - \operatorname{erf}\left(\frac{2^x}{\sigma\sqrt{2}}\right)$  for  $x \in \mathbb{R}$ . If f(x) is unimodal, then the discrete function P(X=x), which is the evaluation of f(x) at integer points, will also be unimodal.

Let  $u = \frac{2^{x^2}}{\sigma\sqrt{2}}$ , so that f(x) = erf(2u) - erf(u). The derivative of the error function is  $\frac{d}{dz}\text{erf}(z) = \frac{2}{\sqrt{\pi}}e^{-z^2}$ . Thus, the derivative of f with respect to x is:

$$\frac{df}{dx} = \frac{2}{\sqrt{\pi}} u \ln 2e^{-u^2} \left( 2e^{-3u^2} - 1 \right)$$

Let  $h(u) = 2e^{-3u^2} - 1$ . Since  $\frac{2}{\sqrt{\pi}}$ , u,  $\ln 2$ , and  $e^{-u^2}$  are all positive for u > 0 (as  $2^x > 0$ ), the sign of  $\frac{df}{dx}$  is determined solely by h(u).

Setting h(u) = 0 gives:

$$2e^{-3u^2} = 1 \implies e^{-3u^2} = \frac{1}{2} \implies -3u^2 = -\ln 2 \implies u^2 = \frac{\ln 2}{3}$$

Thus, the unique critical point is at  $u_0 = \sqrt{\frac{\ln 2}{3}}$ .

For  $u < u_0$ , we have  $3u^2 < \ln 2$ , so  $e^{-3u^2} > \frac{1}{2}$ , meaning h(u) > 0 and  $\frac{df}{dx} > 0$ , so f(x) is increasing.

For  $u > u_0$ , we have  $3u^2 > \ln 2$ , so  $e^{-3u^2} < \frac{1}{2}$ , meaning h(u) < 0 and  $\frac{df}{dx} < 0$ , so f(x) is decreasing.

Therefore, f(x) has a single maximum at  $u_0$ , proving that it is unimodal. Since P(X = x) is the discrete sampling of f(x) at integer values, it follows that P(X = x) is also unimodal.

**Theorem A.2.** Contiguity of Top-K in Unimodal Distributions.

*Proof.* Proof by contradiction: Suppose that the set  $X_K$  of the Top-K most probable values is not contiguous. Then, there exist three integers  $x_a < x_c < x_b$  such that:  $x_a, x_b \in X_K$  but  $x_c \notin X_K$ .

By the unimodal property, the probability function P(x) first increases and then decreases, so for any  $x_c$  between  $x_a$  and  $x_b$ , we have:

$$P(x_c) \ge \min(P(x_a), P(x_b)).$$

Since  $x_a$  and  $x_b$  are in  $X_K$ , they are among the K largest probabilities. Thus,  $\min(P(x_a), P(x_b))$  is at least as large as the K-th largest probability. Therefore,  $P(x_c)$  must also be at least as large as the K-th largest probability, meaning  $x_c$  should be in  $X_K$ .

This contradicts the assumption that  $x_c \notin \mathcal{X}_K$ . Hence, the Top-K set must be contiguous.

#### References

- <span id="page-14-23"></span>[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming throughput-latency tradeoff in LLM inference with sarathi-serve. In Proceedings of the 18th USENIX Conference on Operating Systems Design and Implementation (Santa Clara, CA, USA) (OSDI'24). USENIX Association, USA, Article 7, 18 pages.
- <span id="page-14-6"></span>[2] Mistral AI. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825 (2023).
- <span id="page-14-1"></span>[3] Zeyuan Allen-Zhu and Yuanzhi Li. 2025. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. In ICLR. OpenReview.net.
- <span id="page-14-8"></span>[4] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2024. Quarot: Outlier-free 4-bit inference in rotated llms. arXiv preprint arXiv:2404.00456 (2024).
- <span id="page-14-20"></span>[5] Feng Cheng, Cong Guo, Chiyue Wei, Junyao Zhang, Changchun Zhou, Edward Hanson, Jiaqi Zhang, Xiaoxiao Liu, Hai Li, and Yiran Chen. 2025. Ecco: Improving Memory Bandwidth and Capacity for LLMs via Entropy-Aware Cache Compression. In Proceedings of the 52nd Annual International Symposium on Computer Architecture. 793–807.
- <span id="page-14-2"></span>[6] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. 2024. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. In ICML. OpenReview.net.
- <span id="page-14-17"></span>[7] Esha Choukse, Mattan Erez, and Alaa R Alameldeen. 2018. Compresso: Pragmatic main memory compression. In 2018 51st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 546–558.
- <span id="page-14-18"></span>[8] Esha Choukse, Michael B Sullivan, Mike O'Connor, Mattan Erez, Jeff Pool, David Nellans, and Stephen W Keckler. 2020. Buddy compression: Enabling larger memory for deep learning and hpc workloads on gpus. In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA). IEEE, 926–939.
- <span id="page-14-21"></span>[9] Tri Dao. 2024. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In *International Conference on Learning Representations (ICLR)*.

- <span id="page-14-22"></span>[10] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems (NeurIPS).
- <span id="page-14-12"></span>[11] Rocktim Jyoti Das, Liqun Ma, and Zhiqiang Shen. 2023. Beyond size: How gradients shape pruning decisions in large language models. arXiv preprint arXiv:2311.04902 (2023).
- <span id="page-14-9"></span>[12] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. Advances in Neural Information Processing Systems 35 (2022), 30318–30332.
- <span id="page-14-26"></span>[13] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient Finetuning of Quantized LLMs. In NeurIPS.
- <span id="page-14-13"></span>[14] Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, and Xiaowen Chu. 2024. Pruner-Zero: Evolving Symbolic Pruning Metric from Scratch for Large Language Models. In Proceedings of the 41st International Conference on Machine Learning. PMLR. https://arxiv.org/abs/2406.02924 [arXiv: 2406.02924].
- <span id="page-14-10"></span>[15] Peijie Dong, Lujun Li, Yuedong Zhong, Dayou Du, Ruibo Fan, Yuhan Chen, Zhenheng Tang, Qiang Wang, Wei Xue, Yike Guo, et al. 2024. Stbllm: Breaking the 1-bit barrier with structured binary llms. arXiv preprint arXiv:2408.01803 (2024).
- <span id="page-14-16"></span>[16] Peijie Dong, Zhenheng Tang, Xiang Liu, Lujun Li, Xiaowen Chu, and Bo Li. 2025. Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression. arXiv preprint arXiv:2505.19433 (2025).
- <span id="page-14-0"></span>[17] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
- <span id="page-14-5"></span>[18] Jarek Duda, Khalid Tahboub, Neeraj J Gadgil, and Edward J Delp. 2015. The use of asymmetric numeral systems as an accurate replacement for Huffman coding. In 2015 Picture Coding Symposium (PCS). IEEE, 65–69.
- <span id="page-14-11"></span>[19] Ali Edalati, Alireza Ghaffari, Mahsa Ghazvini Nejad, Lu Hou, Boxing Chen, Masoud Asgharian, and Vahid Partovi Nia. 2025. OAC: Outputadaptive Calibration for Accurate Post-training Quantization. In AAAI. AAAI Press, 16453–16461.
- <span id="page-14-19"></span>[20] Magnus Ekman and Per Stenstrom. 2005. A robust main-memory compression scheme. In 32nd International Symposium on Computer Architecture (ISCA'05). IEEE, 74–85.
- <span id="page-14-14"></span>[21] Ruibo Fan, Xiangrui Yu, Peijie Dong, Zeyu Li, Gu Gong, Qiang Wang, Wei Wang, and Xiaowen Chu. 2025. SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs. In EuroSys. ACM, 243–260.
- <span id="page-14-4"></span>[22] Elias Frantar and Dan Alistarh. 2023. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. In ICML.
- <span id="page-14-3"></span>[23] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323 (2022).
- <span id="page-14-15"></span>[24] Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. 2025. MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models. In PPoPP. ACM, 239–251.
- <span id="page-14-24"></span>[25] Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, and Luo Mai. 2024. ServerlessLLM: Low-Latency Serverless Inference for Large Language Models. In OSDI. USENIX Association, 135–153.
- <span id="page-14-7"></span>[26] Gerasimos Gerogiannis, Stijn Eyerman, Evangelos Georganas, Wim Heirman, and Josep Torrellas. 2025. DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model. In Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture®. 184–200.
- <span id="page-14-25"></span>[27] Ruihao Gong, Shihao Bai, Siyu Wu, Yunqian Fan, Zaijun Wang, Xiuhong Li, Hailong Yang, and Xianglong Liu. 2025. Past-Future Scheduler for LLM Serving under SLA Guarantees. In Proceedings of the 30th

- ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 798–813.
- <span id="page-15-8"></span>[28] Yongchang Hao, Yanshuai Cao, and Lili Mou. 2024. NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks. CoRR abs/2410.20650 (2024).
- <span id="page-15-6"></span>[29] Moshik Hershcovitch, Andrew Wood, Leshem Choshen, Guy Girmonsky, Roy Leibovitz, Ilias Ennmouri, Michal Malka, Peter Chin, Swaminathan Sundararaman, and Danny Harnik. 2024. ZipNN: Lossless Compression for AI Models. CoRR abs/2411.05239 (2024).
- <span id="page-15-29"></span>[30] Connor Holmes, Masahiro Tanaka, Michael Wyatt, Ammar Ahmad Awan, Jeff Rasley, Samyam Rajbhandari, Reza Yazdani Aminabadi, Heyang Qin, Arash Bakhtiari, Lev Kurilenko, and Yuxiong He. 2024. DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference. arXiv[:2401.08671](https://arxiv.org/abs/2401.08671) [cs.PF] [https://arxiv.](https://arxiv.org/abs/2401.08671) [org/abs/2401.08671](https://arxiv.org/abs/2401.08671)
- <span id="page-15-7"></span>[31] David A Huffman. 2007. A method for the construction of minimumredundancy codes. Proceedings of the IRE 40, 9 (2007), 1098–1101.
- <span id="page-15-19"></span>[32] Aaron Jarmusch, Nathan Graddon, and Sunita Chandrasekaran. 2025. Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks. arXiv preprint arXiv:2507.10789 (2025).
- <span id="page-15-9"></span>[33] Jeff Johnson. 2024. DIET-GPU: Efficient Model Inference on GPUs. <https://github.com/facebookresearch/dietgpu>.
- <span id="page-15-14"></span>[34] Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, et al. 2023. Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. In Proceedings of the 50th annual international symposium on computer architecture. 1–14.
- <span id="page-15-12"></span>[35] Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, et al. 2019. A study of BFLOAT16 for deep learning training. arXiv preprint arXiv:1905.12322 (2019).
- <span id="page-15-1"></span>[36] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. CoRR abs/2001.08361 (2020).
- <span id="page-15-15"></span>[37] Hyungyo Kim, Gaohan Ye, Nachuan Wang, Amir Yazdanbakhsh, and Nam Sung Kim. 2024. Exploiting intel advanced matrix extensions (AMX) for large language model inference. IEEE Computer Architecture Letters 23, 1 (2024), 117–120.
- <span id="page-15-25"></span>[38] Jungrae Kim, Michael Sullivan, Esha Choukse, and Mattan Erez. 2016. Bit-plane compression: Transforming data for better compression in many-core architectures. ACM SIGARCH Computer Architecture News 44, 3 (2016), 329–340.
- <span id="page-15-11"></span>[39] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In SOSP. ACM, 611–626.
- <span id="page-15-31"></span>[40] Hoil Lee, Fadhel Ayed, Paul Jung, Juho Lee, Hongseok Yang, and Francois Caron. 2023. Deep Neural Networks with Dependent Weights: Gaussian Process Mixture Limit, Heavy Tails, Sparsity and Compressibility. J. Mach. Learn. Res. 24 (2023), 289:1–289:78.
- <span id="page-15-24"></span>[41] Zhen Li, Yupeng Su, Runming Yang, Zhongwei Xie, Ngai Wong, and Hongxia Yang. 2025. Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning. CoRR abs/2501.03035 (2025).
- <span id="page-15-30"></span>[42] Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving. In OSDI. USENIX Association, 663–679.
- <span id="page-15-3"></span>[43] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and

- Song Han. 2024. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. Proceedings of Machine Learning and Systems 6 (2024), 87–100.
- <span id="page-15-4"></span>[44] Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, and Lu Hou. 2025. Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models. CoRR abs/2504.04823 (2025).
- <span id="page-15-20"></span>[45] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, et al. 2024. Cachegen: Kv cache compression and streaming for fast large language model serving. In Proceedings of the ACM SIG-COMM 2024 Conference. 38–56.
- <span id="page-15-21"></span>[46] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. 2024. SpinQuant–LLM quantization with learned rotations. arXiv preprint arXiv:2405.16406 (2024).
- <span id="page-15-13"></span>[47] Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Qiang Wang, and Xiaowen Chu. 2024. Benchmarking and dissecting the nvidia hopper gpu architecture. In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 656–667.
- <span id="page-15-28"></span>[48] Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, and Lidong Zhou. 2020. Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). USENIX Association, 881–897. [https:](https://www.usenix.org/conference/osdi20/presentation/ma) [//www.usenix.org/conference/osdi20/presentation/ma](https://www.usenix.org/conference/osdi20/presentation/ma)
- <span id="page-15-5"></span>[49] Anmol Mekala, Anirudh Atmakuru, Yixiao Song, Marzena Karpinska, and Mohit Iyyer. 2025. Does quantization affect models' performance on long-context tasks? arXiv preprint arXiv:2505.20276 (2025).
- <span id="page-15-16"></span>[50] NVIDIA. 2020. NVIDIA Ampere GA102 GPU Architecture Whitepaper. [https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) [gpu-architecture-whitepaper-v2.pdf](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf).
- <span id="page-15-17"></span>[51] NVIDIA. 2023. NVIDIA Ada GPU Architecture Whitepaper. [https://images.nvidia.com/aem-dam/Solutions/geforce/ada/](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) [nvidia-ada-gpu-architecture.pdf](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).
- <span id="page-15-18"></span>[52] NVIDIA. 2024. cuBLAS Docs. [https://docs.nvidia.com/cuda/cublas/](https://docs.nvidia.com/cuda/cublas/index.html) [index.html](https://docs.nvidia.com/cuda/cublas/index.html).
- <span id="page-15-10"></span>[53] NVIDIA. 2025. nvcomp: Repository for nvCOMP docs and examples. <https://github.com/NVIDIA/nvcomp>. Accessed: 2025-08-18.
- <span id="page-15-0"></span>[54] OpenAI. 2023. GPT-4 Technical Report. arXiv[:2303.08774](https://arxiv.org/abs/2303.08774) [cs.CL]
- <span id="page-15-22"></span>[55] Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee. 2024. LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models. In ICLR. OpenReview.net.
- [56] Gunho Park, Baeseong Park, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee. 2022. nuQmm: Quantized MatMul for Efficient Inference of Large-Scale Generative Language Models. CoRR abs/2206.09557 (2022).
- <span id="page-15-23"></span>[57] Tommaso Pegolotti, Elias Frantar, Dan Alistarh, and Markus Püschel. 2023. QIGen: Generating Efficient Kernels for Quantized Inference on Large Language Models. CoRR abs/2307.03738 (2023).
- <span id="page-15-26"></span>[58] Gennady Pekhimenko, Vivek Seshadri, Yoongu Kim, Hongyi Xin, Onur Mutlu, Phillip B Gibbons, Michael A Kozuch, and Todd C Mowry. 2013. Linearly compressed pages: A low-complexity, low-latency main memory compression framework. In Proceedings of the 46th Annual IEEE/ACM International Symposium on Microarchitecture. 172–184.
- <span id="page-15-27"></span>[59] Gennady Pekhimenko, Vivek Seshadri, Onur Mutlu, Phillip B Gibbons, Michael A Kozuch, and Todd C Mowry. 2012. Base-delta-immediate compression: Practical data compression for on-chip caches. In Proceedings of the 21st international conference on Parallel architectures and compilation techniques. 377–388.
- <span id="page-15-2"></span>[60] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves

- <span id="page-16-0"></span>to Use Tools. In NeurIPS.
- <span id="page-16-10"></span>[61] Gabin Schieffer, Daniel Araújo De Medeiros, Jennifer Faj, Aniruddha Marathe, and Ivy Peng. 2024. On the rise of amd matrix cores: Performance, power efficiency, and programmability. In 2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 132–143.
- <span id="page-16-22"></span>[62] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. 2024. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. In NeurIPS.
- <span id="page-16-30"></span>[63] Chongjie Si, Jingjing Jiang, and Wei Shen. 2025. Unveiling the Mystery of Weight in Large Foundation Models: Gaussian Distribution Never Fades. CoRR abs/2501.10661 (2025).
- <span id="page-16-25"></span>[64] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. 2024. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles (Austin, TX, USA) (SOSP '24). Association for Computing Machinery, New York, NY, USA, 590–606. doi:[10.1145/3694715.3695964](https://doi.org/10.1145/3694715.3695964)
- <span id="page-16-11"></span>[65] Foteini Strati, Michal Friedman, and Ana Klimovic. 2025. PCcheck: Persistent Concurrent Checkpointing for ML. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1. 811–827.
- <span id="page-16-26"></span>[66] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. 2024. Llumnix: Dynamic Scheduling for Large Language Model Serving. In OSDI. USENIX Association, 173–191.
- <span id="page-16-16"></span>[67] Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2024. A Simple and Effective Pruning Approach for Large Language Models. In ICLR.
- <span id="page-16-8"></span>[68] Gemma Team. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786 (2025).
- <span id="page-16-6"></span>[69] Qwen Team. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 (2024).
- <span id="page-16-1"></span>[70] Qwen Team. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388 (2025).
- <span id="page-16-4"></span>[71] Daniel Waddington and Cornel Constantinescu. 2025. Lossless Compression for LLM Tensor Incremental Snapshots. arXiv preprint arXiv:2505.09810 (2025).
- <span id="page-16-19"></span>[72] Lei Wang, Lingxiao Ma, Shijie Cao, Quanlu Zhang, Jilong Xue, Yining Shi, Ningxin Zheng, Ziming Miao, Fan Yang, Ting Cao, et al. 2024. Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 307–323.
- <span id="page-16-12"></span>[73] Zhuang Wang, Zhaozhuo Xu, Jingyi Xi, Yuke Wang, Anshumali Shrivastava, and TS Eugene Ng. 2025. {ZEN}: Empowering Distributed Training with Sparsity-driven Data Synchronization. In 19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25). 537–556.
- <span id="page-16-2"></span>[74] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chainof-Thought Prompting Elicits Reasoning in Large Language Models. In NeurIPS.
- <span id="page-16-9"></span>[75] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations. 38–45.
- <span id="page-16-23"></span>[76] Mengdi Wu, Xinhao Cheng, Shengyu Liu, Chunan Shi, Jianan Ji, Kit Ao, Praveen Velliengiri, Xupeng Miao, Oded Padon, and Zhihao Jia. 2025. Mirage: A Multi-Level Superoptimizer for Tensor Programs. In 19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25). USENIX Association. [https://www.usenix.org/conference/](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) [osdi25/presentation/wu-mengdi](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)
- <span id="page-16-20"></span>[77] Haojun Xia, Zhen Zheng, Yuchao Li, Donglin Zhuang, Zhongzhu Zhou, Xiafei Qiu, Yong Li, Wei Lin, and Shuaiwen Leon Song. 2023. Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative

- Model Inference with Unstructured Sparsity. Proc. VLDB Endow. 17, 2 (Oct. 2023), 211–224. doi:[10.14778/3626292.3626303](https://doi.org/10.14778/3626292.3626303)
- <span id="page-16-14"></span>[78] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning. PMLR, 38087–38099.
- <span id="page-16-24"></span>[79] Jiarong Xing, Leyuan Wang, Shang Zhang, Jack Chen, Ang Chen, and Yibo Zhu. 2022. Bolt: Bridging the gap between auto-tuners and hardware-native performance. Proceedings of Machine Learning and Systems 4 (2022), 204–216.
- <span id="page-16-17"></span>[80] Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang, Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao, and Ping Luo. 2024. BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation. In ICLR.
- <span id="page-16-3"></span>[81] Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. 2025. Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems. In ICLR. OpenReview.net.
- <span id="page-16-27"></span>[82] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based Generative Models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). USENIX Association, Carlsbad, CA, 521–538. [https://www.usenix.org/conference/](https://www.usenix.org/conference/osdi22/presentation/yu) [osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
- <span id="page-16-7"></span>[83] Patrick Yubeaton, Tareq Mahmoud, Shehab Naga, Pooria Taheri, Tianhua Xia, Arun George, Yasmein Khalil, Sai Qian Zhang, Siddharth Joshi, Chinmay Hegde, and Siddharth Garg. 2025. Huff-LLM: End-to-End Lossless Compression for Efficient LLM Inference. arXiv[:2502.00922](https://arxiv.org/abs/2502.00922) [cs.LG] <https://arxiv.org/abs/2502.00922>
- <span id="page-16-13"></span>[84] Lin Zhang, Longteng Zhang, Shaohuai Shi, Xiaowen Chu, and Bo Li. 2023. Evaluation and optimization of gradient compression for distributed deep learning. In 2023 IEEE 43rd International Conference on Distributed Computing Systems (ICDCS). IEEE, 361–371.
- <span id="page-16-5"></span>[85] Tianyi Zhang, Yang Sui, Shaochen Zhong, Vipin Chaudhary, Xia Hu, and Anshumali Shrivastava. 2025. 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float. arXiv preprint arXiv:2504.11651 (2025).
- <span id="page-16-18"></span>[86] Yingtao Zhang, Haoli Bai, Haokun Lin, Jialin Zhao, Lu Hou, and Carlo Vittorio Cannistraci. 2024. Plug-and-play: An efficient posttraining pruning method for large language models. In The Twelfth International Conference on Learning Representations.
- <span id="page-16-21"></span>[87] Jishen Zhao, Sheng Li, Jichuan Chang, John L Byrne, Laura L Ramirez, Kevin Lim, Yuan Xie, and Paolo Faraboschi. 2015. Buri: Scaling bigmemory computing with hardware-based memory expansion. ACM Transactions on Architecture and Code Optimization (TACO) 12, 3 (2015), 1–24.
- <span id="page-16-15"></span>[88] Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. 2024. Atom: Low-bit quantization for efficient and accurate llm serving. Proceedings of Machine Learning and Systems 6 (2024), 196–209.
- <span id="page-16-28"></span>[89] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2025. SGLang: efficient execution of structured language model programs. In Proceedings of the 38th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS '24). Curran Associates Inc., Red Hook, NY, USA, Article 2000, 27 pages.
- <span id="page-16-29"></span>[90] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). USENIX Association, Santa Clara, CA, 193– 210. [https://www.usenix.org/conference/osdi24/presentation/zhong](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)[yinmin](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)