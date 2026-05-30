## veRL Framework (Volcano Engine Reinforcement Learning, 火山引擎强化学习框架)

术语解释
veRL (Volcano Engine Reinforcement Learning) 是字节跳动/火山引擎开源的大规模 LLM 强化学习训练框架（https://github.com/volcengine/verl），支持 PPO、GRPO、DPO 等 RLHF 算法。其 Fully-Sharded Data Parallel (FSDP) Trainer 被 CoE 论文用于模型训练，并扩展支持 multi-round expert execution 和 fine-grained token-level logging。

术语是什么？
veRL 是一个灵活的 RLHF 训练框架，基于 Ray 实现分布式训练，支持混合并行策略（Data Parallelism + Tensor Parallelism + Pipeline Parallelism）。核心组件包括：
- **FSDP Trainer**：使用 PyTorch FSDP 进行分片训练，支持 ZeRO 优化
- **HybridFlow**：灵活的计算-通信编排，支持将不同 RL 阶段（rollout、training、reward）分配到不同硬件资源
- **vLLM Integration**：支持 vLLM 进行 rollout 阶段的快速推理

CoE 论文使用 veRL 的 FSDP Trainer 作为基础训练框架，并扩展以支持：(1) CoE layer 内多次 expert 前向（multi-round expert execution）；(2) 每步 Router 的独立参数管理和梯度更新；(3) 细粒度的 token-level routing 日志。

从系统架构角度拆解术语：
veRL 的 FSDP Trainer 训练流程：
1. 数据加载：从 dataset 加载 batch → 分发到各 GPU rank
2. FSDP Sharding：模型参数按 FSDP unit 分片到各 GPU
3. Forward：各 GPU 执行本地 shard 的前向计算 → all-gather 需要的参数 → MoE layer 执行 all-to-all dispatch/combine
4. Backward：梯度计算 → reduce-scatter 梯度
5. Optimizer Step：各 GPU 更新本地参数 shard

CoE 的修改：在 FSDP Trainer 的 forward 函数中，将单次 MoE forward 替换为 C 次迭代的 MoE forward（CoE forward），每步执行独立的 Router → TopK → Expert FFN → Residual Add。

术语一般如何实现？如何使用？
- 安装：pip install verl 或从源码编译（https://github.com/volcengine/verl）
- 训练脚本：使用 YAML 配置文件指定模型架构、并行策略、训练超参数
- 适用场景：LLM 的 RLHF/RL 训练（PPO、GRPO、DPO），也可作为通用分布式训练框架（如 CoE 的 SFT 训练）
- 关键特性：支持 FSDP、TP、PP 混合并行，与 vLLM 集成，灵活的资源分配

涉及论文标题：
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models
