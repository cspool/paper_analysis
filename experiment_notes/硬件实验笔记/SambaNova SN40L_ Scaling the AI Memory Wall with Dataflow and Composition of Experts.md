## SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts

- 属于硬件架构的实现是什么？实验比较什么？
  实现：SN40L Reconfigurable Dataflow Unit (RDU)，一款商业数据流加速器芯片。TSMC 5nm工艺，2.5D CoWoS chiplet封装，每socket含两个RDD+ HBM。核心组件：(1) PCU — 1040个，可配置为output stationary systolic array（GEMM）或pipelined SIMD core（element-wise操作），支持BF16/FP32/INT32，含cross-lane reduction network和tail transcendental functions；(2) PMU — 1040个，含scratchpad SRAM + 标量ALU流水线用于并发读写地址生成，data alignment unit支持in-place transpose，可编程bank bits消除bank冲突；(3) AGCU — RDU tile到HBM/DDR的桥接单元，支持P2P inter-RDU通信和硬件kernel launch编排；(4) RDN — 2D mesh互连，含vector/scalar/control三种fabric，支持multicast、静态流路由、sequence ID based数据重排序，credit-based per-hop流控；(5) 三级存储体系：520 MiB片上PMU SRAM + 64 GiB co-packaged HBM (~1.8 TB/s) + 最高1.5 TiB DDR DRAM (~200 GB/s)。
  实验比较：在8-socket SN40L Node上，对比Unfused（每个PyTorch operator独立kernel执行，中间结果materialize）、Fused+Software Orchestrated（编译器自动融合+host软件调度kernel）、Fused+Hardware Orchestrated（编译器自动融合+AGCUs硬件调度kernel）三种配置；以及SN40L Node vs DGX A100 vs DGX H100在Samba-CoE推理场景下的延迟和系统占用。

- 硬件平台是什么，配置是什么。
  SN40L RDU：638 BF16 TFLOPS，1040 PCU + 1040 PMU，520 MiB片上SRAM，64 GiB HBM（~1.8 TB/s），1.5 TiB DDR（~200 GB/s），时钟<2 GHz，die size <650 mm²。实验用8-socket SN40L Node（Samba-CoE实验，DDR→HBM聚合带宽>1 TB/s），16-socket（Llama 3.1推理），单socket（FlashFFTConv benchmark）。对比平台：DGX A100（8×A100 80GB PCIe，32 GB/s host-to-GPU带宽），DGX H100（8×H100 80GB，64 GB/s host-to-GPU带宽）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama2-7B（prefill/decode/train）、Llama2-70B（prefill/decode）、sparseGPT-13B（87.5%稀疏训练）、Bloom-176B（prefill/decode）、Mistral-7B（prefill/decode）、Falcon-40B（prefill/decode）、LLaVA1.5-7B multimodal（prefill/decode）、FlashFFTConv（1M序列长度FFT卷积）、Llama3.1-8B/70B/405B（推理）。Samba-CoE：150个Llama2-7B experts + 1个router model，超1T总参数。Benchmark方面，模型性能指标聚焦吞吐量（tokens/s）和延迟（ms），未使用传统NLP accuracy benchmark进行评测。

- 开源情况。基于开源文档和论文，使用例子解释，解释硬件架构。
  论文未开源（SN40L为SambaNova商业芯片）。硬件架构的执行原理：编译器将PyTorch模型算子图编译为空间融合dataflow kernel，映射到PCU/PMU/AGCU/RDN。以Figure 4所示GEMM→Mul→Transpose→GEMM融合为例：Gemm0映射到多个PCU配为systolic array，中间结果通过RDN流式写入PMU stage buffer I0（partition为I00/I01以匹配带宽），Mul在另一组PCU上以SIMD模式执行，Transpose通过PMU data alignment unit的diagonally striped write + normal read实现（无需显式数据搬移），Gemm1在更多PCU上以systolic执行。PMU的scalar ALU pipeline并发生成读写地址，sequence ID用于many-to-one数据重排序。AGCUs从HBM流式加载权重/激活、写回结果。整个decoder layer被编译为单个kernel launch，PCU和PMU利用率~90%，HBM带宽利用率~85%。
