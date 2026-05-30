## A Survey on Inference Optimization Techniques for Mixture of Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  本论文是综述，不提供原始实验。它系统性地分类和比较了 MoE 推理优化中的系统级技术（Section 4）：
  - **Expert Parallelism（4.1）**：
    - 并行策略设计：Tutel（动态切换并行策略）、Alpa（intra/inter-operator并行分类）、DeepSpeed-TED（data+tensor+expert混合并行）、BaGuaLu（MoDa策略）、SmartMoE（异构感知混合并行）、MPMoE（pipeline并行优化）
    - 负载均衡：Prophet（性能建模+greedy搜索）、MoE-Prediction（预测专家负载）、Lazarus（专家副本分配）、FlexMoE（细粒度复制）、Brainstorm（历史分配数据）、Lynx（减少batch中的激活专家）、BaseLayers（线性分配问题）、MoE-ECR（expert选择token）
    - All-to-All通信优化：Tutel/HetuMoE/DeepSpeed-MoE（分层all-to-all）、TA-MoE/DeepSpeed-TED（数据压缩）、Janus（以数据为中心移动expert）、ExFlow（减少all-to-all操作数）、Aurora（有序token传输避免带宽竞争）、LocMoE/Parm（inter-node转intra-node）
    - 任务调度：ScMoE（shortcut架构解耦通信）、HiDup（microbatch重叠通信与计算）、MoESys（2D预取+融合通信）、ScheMoE（模块化+自适应调度）、PipeMoE（性能建模+pipeline调度）、EPS-MoE（动态kernel选择重叠FFN与通信）
  - **Expert Offloading（4.2）**：
    - 预取：Mixtral-Offloading/AdapMoE/HOBBIT（基于当前门控输入预测下层expert）、Pre-gated MoE（预门控结构）、EdgeMoE（预测表）、DyNN-Offload（pilot模型预测）、MoE-Infinity（请求级频率追踪）、ProMoE（学习型预测器滑窗预取）、ExpertFlow/SiDA（一次性预测所有expert）
    - 缓存：LRU策略（Mixtral-Offloading等）、LFU策略（MoE-Infinity）、静态重要性配置（Fiddler）、动态缓存更新（SwapMoE）、动态缓存大小（AdapMoE）、多维策略+混合精度（HOBBIT的LRU+LFU+LHU）、cache-aware routing（CacheMoE）
    - 加载优化：低精度expert加载（EdgeMoE、HOBBIT）、自适应跳过不重要expert（AdapMoE）
    - CPU辅助：Fiddler（CPU执行expert计算）、HOBBIT（CPU处理低精度expert）、MoE-Lightning（CPU-GPU-I/O流水线）
  
  表5-10汇总了各系统的加速比、内存节省、GPU利用率等性能指标。

- 硬件平台是什么，配置是什么。
  - 云集群场景：多GPU服务器（如NVIDIA A100/H100），支持分布式训练和推理
  - 边缘设备场景：单GPU内存受限设备，部分使用Jetson Orin等嵌入平台
  - 论文未统一规定硬件配置，各被综述系统使用各自的实验配置

- 开源Serving框架是什么。修改了什么。
  **开源框架基础**（Table 10统计）：
  - PyTorch（12个并行系统+4个offloading系统基于此）
  - DeepSpeed（9个并行系统+1个offloading系统）
  - Transformers（5个并行系统+7个offloading系统）
  - 其他：Fairseq、Llama.cpp、vLLM、FasterTransformer各1-2个系统
  
  **主要修改方向**：
  - DeepSpeed-MoE：在DeepSpeed上增加MoE支持，包括分层all-to-all、expert parallelism
  - Tutel：在Fairseq/PyTorch上实现自适应并行策略切换
  - vLLM-based：利用PagedAttention管理expert参数的KV cache

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **开源情况**：综述关联仓库 https://github.com/MoE-Inf/awesome-moe-inference/ 汇总各系统的开源链接。主要开源系统包括：
  - DeepSpeed-MoE: https://github.com/microsoft/DeepSpeed
  - Tutel: https://github.com/microsoft/tutel
  - FasterMoE: https://github.com/laekov/fastermoe
  - vLLM: https://github.com/vllm-project/vllm

  **Expert Offloading框架执行全过程（以Mixtral-8x7B + HOBBIT为例）**：
  1. 输入：用户请求tokens到达GPU
  2. 非expert参数（Attention、Router、LayerNorm）常驻GPU显存
  3. Expert Cache：GPU显存中缓存高频expert（FP16高精度）
  4. Router计算：对每个token计算θ = Softmax(R(x))，选出top-K expert
  5. Cache查询：检查所需expert是否在GPU expert cache中
  6. Cache Miss处理：
     a. 计算该expert的importance score（基于gate输出）
     b. 若score低于阈值 → 从CPU/SSD加载低精度版本（INT4）
     c. 若score高于阈值 → 加载高精度版本（FP16）
     d. 同时触发下层expert预取（基于当前gate输出预测下层）
  7. Expert计算：GPU对已加载expert执行FFN计算
  8. 加权聚合：合并expert输出
  9. 输出：生成下一个token

  **Expert Parallelism框架执行全过程（以DeepSpeed-MoE为例）**：
  1. 输入：batch tokens分布在各GPU上
  2. 每GPU持有部分expert + 全部非expert参数
  3. Attention + Router计算（本地）
  4. All-to-All通信：根据Router输出将token分发到持有对应expert的GPU
  5. Expert计算（本地GPU对分配到的token执行FFN）
  6. All-to-All通信：将expert输出传回原始GPU
  7. 继续下一层
