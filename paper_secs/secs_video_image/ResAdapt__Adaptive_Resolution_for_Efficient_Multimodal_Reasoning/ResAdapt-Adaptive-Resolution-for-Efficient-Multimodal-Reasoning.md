# ResAdapt: Adaptive Resolution for Efficient Multimodal Reasoning

**Huanxuan Liao***τ***,** *<sup>µ</sup>* **, Zhongtao Jiang, Yupu Hao***τ***,** *<sup>µ</sup>* **, Yuqiao Tan***τ***,** *<sup>µ</sup>* **, Shizhu He***τ***,** *<sup>µ</sup>* **, Ben Wang, Jun Zhao***τ***,** *<sup>µ</sup>* **, Kun Xu**† **, Kang Liu***τ***,** *<sup>µ</sup>***,** <sup>∗</sup>

Institute of Automation, Chinese Academy of Sciences , *<sup>µ</sup>*University of Chinese Academy of Sciences ,

†Project Leader <sup>∗</sup>Corresponding author: [kliu@nlpr.ia.ac.cn](mailto:kliu@nlpr.ia.ac.cn)

Scaling both spatial resolution and temporal coverage in video reasoning demands visual-token budgets that grow prohibitively for Multimodal Large Language Models (MLLMs). Existing efficiency strategies intervene too late: model-side token pruning discards fine-grained evidence after the encoder has already paid the full computational cost, while output-side iterative retrieval introduces multi-turn latency. We propose **ResAdapt**, a framework that reallocates visual budget *before* encoding. A lightweight, query-aware Allocator predicts a per-frame resolution scale, adjusting the pixels the backbone receives while preserving its native token interface and compatibility with optimized inference engines. To train this non-differentiable pipeline, we introduce **Cost-Aware Policy Optimization (CAPO)**, which combines a dynamic cost pivot with asymmetric reward shaping to jointly maximize reasoning accuracy under strict visual budgets—preventing the policy collapse that plagues direct cost penalties. The resulting Allocator concentrates pixels on information-dense frames, exhibiting content-adaptive active perception learned entirely from task reward. Across video QA and temporal grounding benchmarks, ResAdapt matches or exceeds uncompressed baselines while eliminating over 90% of visual tokens. Crucially, the saved spatial budget is reinvested into temporal coverage: under equivalent compute, ResAdapt processes 16× more frames, yielding > 15% relative gains on complex long-video reasoning tasks.

**Project Page**: <https://xnhyacinth.github.io/projects/ResAdapt> **Code Repository**: <https://github.com/Xnhyacinth/ResAdapt>

**Contact**: [liaohuanxuan2023@ia.ac.cn](mailto:liaohuanxuan2023@ia.ac.cn)

