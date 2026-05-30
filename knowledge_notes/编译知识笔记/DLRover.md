## DLRover

术语解释
DLRover 是蚂蚁集团开源的大规模分布式深度学习训练系统（https://github.com/intelligent-machine-learning/dlrover），Ling 模型的训练基础设施核心。统一管理 DeepSpeed、Megatron-LM 及其 vendor 版本跨异构硬件平台部署，内置 XPUTimer 性能分析和 EDiT 异步训练策略。

术语是什么？
DLRover 提供一个跨平台训练抽象层，解决异构加速器（DSA 和 GPGPU）环境中的底层算子不一致问题：(1) 与硬件厂商合作标准化底层算子（group_gemm、permute/unpermute、all2all）并确保计算和通信一致性；(2) 开发跨平台兼容层无缝集成多个分布式训练框架；(3) 实现高效调试机制快速定位异构环境中的问题。

内置组件包括：XPUTimer（轻量级性能诊断）、EDiT（弹性异步训练）、PCache（分布式存储缓存）、Babel（跨集群数据同步）。

从编译框架角度拆解术语：
```
=== DLRover 训练启动流程 ===
$ dlrover run --framework=megatron --platform=device_D \
    --model=ling_plus --devices=1000 \
    --distributed="tp=8 ep=64 pp=4" \
    --storage="pcache://cluster_a" \
    --profiler="xputimer"

# 内部: 将 Megatron training script 适配到目标平台
# - 替换 device-specific kernel (cuBLAS→vendor BLAS)
# - 注入 XPUTimer tracing hooks
# - 配置 EDiT 异步同步策略
```

术语一般如何实现？如何使用？
- 开源：https://github.com/intelligent-machine-learning/dlrover
- 支持训练框架：DeepSpeed, Megatron-LM, Megatron vendor versions
- 跨平台：支持 5 种异构 AI 加速器的无缝切换
- 训练任务可通过统一命令行接口在不同集群间迁移

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs
