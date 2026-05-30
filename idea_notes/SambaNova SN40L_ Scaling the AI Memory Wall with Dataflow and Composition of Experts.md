## SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts

- baseline方法是什么？
  Baseline方法：(1) **GPU上的传统AI加速器架构**（NVIDIA DGX A100 / DGX H100）：不具备三级存储体系（仅有HBM+host DRAM via PCIe），无法高效部署Composition of Experts系统。GPU operator fusion受限于rigid memory hierarchy（SM间仅通过shared cache/HBM交换数据）、on-chip SRAM容量不足、SIMT编程模型不支持跨算子的pipeline parallelism。全栈执行例子（CoE推理在DGX H100上）：prompt请求→ router在GPU HBM中执行→需切换expert时从host DRAM通过PCIe拷贝权重（DGX H100: 64 GB/s）→expert在GPU上以多个独立kernel执行（prefill/attention/MLP各为独立kernel launch，中间结果materialize到HBM）→自回归decoding每次迭代重复加载权重和KV cache→超过50个expert时HBM溢出到host DRAM导致切换延迟剧增→DGX在150 experts时OOM。
  全栈执行例子（Baseline: DGX H100 CoE推理，single token decode）：
  ```
  # 算法层：标准Llama2-7B decoder layer，attention + FFN
  # 系统框架层：PyTorch/TensorRT，每个operator→独立CUDA kernel
  # GPU执行：HBM→L2→SM→HBM，intermediate materialization
  # 编译框架层：TensorRT operator fusion（1-5 operators，access pattern受限）
  # Kernel调度层：CUDA kernel launch开销 + software orchestration
  # 硬件架构层：H100 HBM (3.35 TB/s) + host DRAM via PCIe (64 GB/s)
  # 缺陷：
  # 1. 无法融合含transpose的arbitrary access pattern → 操作强度低（39.5 Ops/Byte），memory-bound
  # 2. HBM容量有限（80 GB），150个7B experts无法全部驻留
  # 3. 模型切换走PCIe（64 GB/s），延迟高（Figure 1）
  # 4. Decode kernel短，CUDA kernel launch开销占比高
  # 5. 无pipeline parallelism → 小矩阵乘法（32×32×32）无法充分利用所有SM
  ```

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**SN40L Reconfigurable Dataflow Unit + 三级存储体系 + Streaming Dataflow + Samba-CoE**。核心设计：(1) Streaming dataflow — coarse-grained pipeline执行算子图，tensor被tiled并streaming通过PCU/PMU pipeline，transpose等复杂access pattern通过PMU data alignment unit在buffer内实现（in-place），无需HBM materialization；(2) 三级存储体系 — 520 MiB片上SRAM(PMU) + 64 GiB HBM + 1.5 TiB DDR，CoE专家参数存DDR、活跃专家拷贝到HBM（聚合带宽>1 TB/s vs GPU 32-64 GB/s）、Router+KV cache常驻HBM；(3) 自动空间融合 — 编译器将整个decoder layer融合为单个kernel调用，20+ operators在一个kernel内完成；(4) 硬件orchestrated kernel launch — AGCUs硬件调度kernel序列，消除host software调度开销，对短执行时间的decode kernel效果显著（1.4×-8× speedup）。
  全栈执行例子（对比baseline，SN40L CoE推理，single token decode）：
  ```
  # 算法层：相同Llama2-7B decoder layer（不改模型架构）
  # 系统框架层：Samba-CoE Runtime — Router常驻HBM，expert权重按需 DDR→HBM>1 TB/s
  # → CoE Runtime LRU管理HBM中活跃expert，读多写少weight跳过回写
  # 编译框架层：SambaNova编译器自动融合 — 将PyTorch算子图编译为空间融合dataflow kernel
  # → 静态符号生命周期分析实现garbage collection
  # → 静态带宽建模控制RDN/TLN并发流资源分配
  # → Place-and-Route配置RDN路由、flow ID、multicast路径
  # Kernel调度层（硬件orchestrated）：单个Kernel Execute命令发出后
  # → AGCUs硬件执行静态kernel schedule序列，无需host参与
  # → HBM→AGCU→RDN→PCU(systolic GEMM for QKV)→PMU(stage buffer)→PCU(SIMD activation)→PMU→...
  # → 整个decoder layer在一个kernel内完成，无中间结果materialize
  # 硬件架构层：PCU(systolic/SIMD双模)+PMU(scratchpad+地址生成+data alignment)+RDN(mesh+flow control)
  # → PMU data alignment unit: transpose通过diagonally striped write实现（无需数据搬移）
  # → RDN credit-based flow control + sequence ID重排序
  # → AGCU桥接片上RDN与片外HBM/DDR
  # → Die-to-die接口: 两个RDD间直接流式传输
  # → Peer-to-peer: 跨socket direct RDU通信（AllReduce等collective fused进kernel）
  # 
  # 解决Baseline缺陷：
  # 1. 操作强度：39.5→410.4 Ops/Byte（全面融合），FlashFFTConv 13× speedup
  # 2. 模型切换延迟：DDR→HBM >1 TB/s vs PCIe 32-64 GB/s → 15×-31× 切换加速
  # 3. HBM带宽利用率：~85%（dataflow重叠weight load与compute）vs GPU ~50%
  # 4. Kernel launch开销：硬件orchestrated消除host往返（decode阶段1.4×-8× speedup）
  # 5. System footprint：单Node支持850 experts (TP8) vs DGX需19节点 → 19× footprint缩减
  # 6. 150+ experts: DGX OOM, SN40L正常服务
  # 7. 小模型pipeline parallelism：PCU间chaining实现pipeline，利用多个小GEMM的并行性
  ```
