## Cold Start Mitigation in Serverless LLM Inference（Serverless LLM 推理冷启动缓解）

术语是什么？
Cold Start 是 serverless 计算的核心性能问题：当 serverless function 被首次调用或长时间未使用后，平台需要分配计算资源、加载容器镜像、初始化运行时环境和模型参数，这段时间称为冷启动延迟。在 LLM 推理场景中，冷启动尤为严重——模型参数可达数十 GB，加载到 GPU/CPU 内存需要数秒至数十秒。

Remoe 针对 MoE serverless 推理提出三重冷启动缓解策略：
1. **主模型与 Remote Experts 并行冷启动**：将低频 experts 移出主模型容器，减小主模型镜像大小和参数加载量；remote expert Pods 的冷启动与主模型 Pod 冷启动并行进行
2. **MMP 预分配与 SPS 预测重叠**：资源预分配计算（MMP）在 pre-processing 层冷启动期间完成，不额外增加等待时间
3. **减少单容器内存需求**：通过 heterogeneous 架构（GPU: 非 expert + CPU: local experts）+ remote experts 分离，主模型容器内存需求显著降低 → 更快的模型加载

从系统架构角度拆解术语：
Remoe 冷启动时间分解：
```
Total Cold Start = max(
    T_container_startup_base,           // 所有方法共享的容器基础启动时间（相同 base image）
    T_main_model_load + T_optimization, // 主模型加载 + MMP/Lagrangian/LPT 计算
    T_remote_experts_cold_start         // Remote expert Pods 冷启动（可与主模型重叠）
)
其中 T_optimization ≈ 0（CALCULATE 开销可忽略）
```
Remoe 实现 47% 冷启动时间减少（vs baselines），主要收益来自：主模型镜像不含低频 experts → 更小的下载/加载量 + remote experts 冷启动与主模型并行。

术语一般如何实现？如何使用？
- ServerlessLLM (OSDI '24)：通过 pipeline parallelism 和多层本地存储加速模型加载
- ParaServe：利用多节点并行加载模型分片
- Remoe 的独特性：通过 expert 分区减少单函数内存需求——不依赖更快的存储或网络，而是从模型架构层面降低冷启动数据量

涉及论文标题：
- Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing
