## Chakra 执行轨迹 / 执行图（Execution Trace / Execution Graph，MLCommons Chakra schema）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Execution Graph（EG）是以计算/通信算子为节点、数据/控制依赖为边的 DAG，编码执行顺序、关键路径、算子重叠与瓶颈；Execution Trace（ET）是 EG 加上真实系统执行后的时序/显存等元数据。Chakra 是 MLCommons 标准化的执行轨迹 schema（github.com/mlcommons/chakra，Chakra Working Group）：用 protobuf 序列化分层 DAG（.et 文件），节点=算子/张量、边=依赖，每 rank 一个文件，配合 comm_group 描述并行通信组。vault 证据：paper_secs 中本论文 A.-Methodology.md（Chakra trace parser、ASTRA-Sim 回放，score 31.0）、IV.-STAGE（默认输出 Chakra schema，28.7）、I.-INTRODUCTION（MLCommons Chakra PyTorch 支持 post-execution 采集，25.9）；RoCC 论文用 STG 工具生成 Chakra trace 并改写执行图做细粒度重叠（30.8/30.7）。ASTRA-sim 条目（知识库_系统架构）也已记录 Chakra ET 作为其 workload 输入。
- 从编译框架角度拆解：Chakra 是分布式 workload "编译器"（如 STAGE）的输出格式——编译产物携带每个节点的算子类型、tensor 尺寸、通信量与依赖关系，下游模拟器（ASTRA-Sim）无需真实集群即可回放评估。STAGE 默认输出 Chakra schema（v0.0.4，另有 JSON backend），并把 Chakra trace 作为与真实 trace 对齐的基准：用 Chakra trace parser 解析、在 ASTRA-Sim 中回放做内存/通信/端到端验证。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 真实 ET 由 PyTorch profiler/Kineto/CUPTI/PARAM/Chakra 在运行真实分布式训练时采集（如 NeMo 24.07 修改集成 PyTorch profiling 采集 Chakra trace）；合成 ET 由 STAGE 等生成器直接产出。使用链：采集或合成 Chakra ET → 模拟器（ASTRA-Sim/SimAI/ScaleSim/Genie）解析回放 → 输出 runtime/显存/通信量/重叠分析 → 指导并行策略选择与硬件 DSE。STAGE 生成 540B 模型 32K GPU trace 约 28 分钟、内存<400 MB，远快于真实采集（LLaMA-3.1-70B 128 micro-batch @32 H100 约 47 GPU-分钟）。

涉及论文标题：
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
