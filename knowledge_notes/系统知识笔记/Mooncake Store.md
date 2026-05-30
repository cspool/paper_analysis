## Mooncake Store

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mooncake Store 是 Moonshot AI 的 Mooncake 项目中的分布式 KVCache 存储引擎，被 EPD-Serve 用作底层跨阶段异步 tensor 传输中间件。它是面向 PD 分离架构的分层缓存系统，支持跨 DRAM、VRAM、NVMe 的批量数据传输，原生支持 TCP、RDMA（InfiniBand/RoCEv2/GPUDirect）、CXL/共享内存、NVMe over Fabric 等协议。Mooncake 的核心设计理念是"用更多存储换取更少计算"（Trading More Storage for Less Computation），获 FAST 2025 最佳论文奖。开源地址：https://github.com/kvcache-ai/Mooncake。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

EPD-Serve 使用 Mooncake Store 的两个关键场景：

```
1) E-P 异步特征预取:
   Encode 实例                   Mooncake Store              Prefill 实例
   put(hash, V_m) ────────────►  store in DRAM/NVMe  ◄──── get(hash)
                                 返回 V_m ────────────────► 写入本地缓存

2) P-D 分层 KV 传输:
   Prefill 实例                  Mooncake Store              Decode 实例
   每层 KVCache ──────────────►  RDMA 传输 ──────────────► 分层接收+拼装
   分组打包 + 延迟调度            拓扑感知路径选择
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Mooncake Store 由三部分构成：(1) Transfer Engine——拓扑感知的多协议传输引擎，40GB KVCache（≈LLaMA3-70B 128K tokens）在 8×400Gbps RoCE 下可达 190 GB/s 传输带宽；(2) Store——分布式缓存池，支持多副本、大对象条带化、并行 I/O，已集成到 vLLM、SGLang、LMCache、TensorRT-LLM 等主流框架；(3) Conductor——KVCache 感知的全局调度器。Mooncake 2025 年的关键里程碑包括：vLLM-Ascend 集成 (2025.08)、SGLang 正式支持 Mooncake Store (2025.09)、TensorRT LLM 集成 (2025.12)。EPD-Serve 利用 Mooncake 的传输接口实现 Ascend NPU 间的 E-P 特征传输和 P-D KV 传输，但论文未详述具体集成方式。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
