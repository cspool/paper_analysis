## 调度IR（mapped IR / operator-scheduled IR，算子-硬件映射中间表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 调度 IR 是硬件模拟/调度器在"算子图 → 可执行调度"之间引入的中间表示，解耦不同调度阶段。NeRArch-Sim 的模块化数据流调度器定义了两级 IR：(1) **mapped IR**——mapping engine 的输出，把算子图与硬件配置结合，包含五段：Operator Information（op_id/op_type/input/output tensors 形状与类型）、Taxonomy-Specific Attributes（Encoding 的 encoding_type/hash_table_size/feature_dim，Field Comp 的 network_depth/hidden_dim/activation，Sampling 的 num_samples/sampling_strategy，Blending 的 blend_mode/accumulation_type）、Hardware Binding Information（算子-硬件绑定）、Resource Requirements、Optimization Techniques；(2) **operator-scheduled IR**——operator-level scheduler 在 mapped IR 上追加执行细节：start_cycle、duration（周期）、资源分配、数据搬移调度、优化元数据，同时保持算子类型无关接口供 system-level scheduler 消费。这与编译领域通用 IR 概念（如 FIRRTL、MLIR、Cross-block IR 等本库条目）不同：它是调度器的输入输出契约而非代码生成中间层。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（对应 NeRArch-Sim CLI 分步命令）：`./nerarch_sim map <execution_dag.pkl> <hardware_config.json> -o mapped_ir.json`（输出 mapped IR）→ `./nerarch_sim schedule mapped_ir.json --hardware <config> -o scheduled_ir.json`（operator-level 调度消费 mapped IR，追加 start_cycle/duration，产出 operator-scheduled IR；可选 --no-ppa）→ `./nerarch_sim report scheduled_ir.json --format html`（system-level 调度 DAGS 消费 operator-scheduled IR 生成 PPA）。示例（GSCore tile 化 3DGS）：mapped IR 中 CULL_AND_CONVERT 算子绑定 CCU、排序算子绑定 QSU/BSU、混合算子绑定 VRU，operator-scheduled IR 记录每算子的 start_cycle/duration（表 VI：CCU 128 cycle、BSU 4 cycle、QSU 64 cycle、VRU 192 cycle），system-level 据此按依赖与资源约束排全局执行计划。IR 字段变化：加优化（tile culling）时 s_comp/r_bytes 因子进入 duration 计算（式 1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：JSON 序列化（mapped_ir.json / scheduled_ir.json），调度器 C++/Python 实现；mapping engine 按统一分类学匹配算子与硬件（多候选取最高吞吐、多实例均衡避免瓶颈、不匹配报 mismatch）。使用价值：三层调度通过 IR 解耦，operator 细节变化不影响 system-level 调度（保持算子类型无关）；IR 直接驱动 PPA/延迟/访存报告与 Gantt 可视化，也可被 `validate` 校验（DAG 无环、四阶段覆盖/顺序）。调度延迟（表 XI）：mapping 2.0~5.3s、op-level 7.1~24.1s、sys-level 1.0~21.1s。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
