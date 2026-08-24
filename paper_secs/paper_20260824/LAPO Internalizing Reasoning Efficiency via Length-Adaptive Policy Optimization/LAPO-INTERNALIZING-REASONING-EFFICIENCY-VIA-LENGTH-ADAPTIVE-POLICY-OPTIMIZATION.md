# LAPO: INTERNALIZING REASONING EFFICIENCY VIA LENGTH-ADAPTIVE POLICY OPTIMIZATION

Xingyu Wu<sup>1</sup> , Yuchen Yan<sup>1</sup> , Shangke Lyu<sup>1</sup> , Linjuan Wu<sup>1</sup> , Yiwen Qiu<sup>1</sup> , Yongliang Shen<sup>1</sup> , Weiming Lu<sup>1</sup> , Jian Shao<sup>1</sup> , Jun Xiao<sup>1</sup> , Yueting Zhuang<sup>1</sup> <sup>1</sup>Zhejiang University {wuxingyu, syl}@zju.edu.cn

§ GitHub: <https://github.com/zju-real/lapo> Project: <https://zju-real.github.io/lapo>

### ABSTRACT

Large reasoning models have achieved remarkable performance through extended chain-of-thought sequences, yet this computational freedom leads to excessive token generation even for simple problems. We present Length-Adaptive Policy Optimization (LAPO), a novel framework that transforms reasoning length control from an external constraint into an intrinsic model capability. Unlike existing approaches that impose rigid limits or rely on post-hoc interventions, LAPO enables models to internalize an understanding of appropriate reasoning depth through a two-stage reinforcement learning process. In the first stage, models learn natural reasoning patterns by discovering the statistical distribution of successful solution lengths. The second stage leverages these patterns as metacognitive guidance, embedding them directly within the model's reasoning context to ensure inference-time flexibility. Experiments on mathematical reasoning benchmarks demonstrate that LAPO reduces token usage by up to 40.9% while improving accuracy by 2.3%. Our analysis reveals that models trained with LAPO develop emergent abilities to allocate computational resources based on problem complexity, achieving efficient reasoning without sacrificing quality.

