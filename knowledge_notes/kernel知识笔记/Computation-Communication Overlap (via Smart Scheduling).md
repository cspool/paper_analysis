## Computation-Communication Overlap (via Smart Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Computation-Communication Overlap 是通过在独立 CUDA stream 上并行执行计算 kernel 和通信操作来隐藏通信延迟的系统级优化技术。FasterMoE 将其实现为 Smart Scheduling 策略：将粗粒度 all-to-all 通信拆分为 n 个 fine-grained 操作序列，在 comm stream 和 comp stream 上重新排列 S（send）、C（compute）、R（receive）操作，尊重数据依赖的同时最大化并行度。核心思想来自 DDL-Roofline 分析的结论——同步执行（半理想曲线）下 end-to-end 延迟 = Lat_comp + Lat_comm，而通过重叠执行可逼近理想曲线（P̄_ideal = P_w · min{1, R_CC}）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Smart Scheduling: 两 stream 调度 (n groups, worker i)
# 数据依赖: C_{i,j} 等 S_{i,j} 完成, R_{i,j} 等 C_{i,j} 完成

# Timeline (图 8b/c 示意, n=4):
# Comm stream:  |S0|S1|S2|S3|     |R0|R1|R2|R3|
# Comp stream:      |C0|C1|C2|C3|    (Cx等对应Sx完成)

# 对比同步执行 (图 8a):
# |S0|S1|S2|S3|C0|C1|C2|C3|R0|R1|R2|R3|

# 延迟分析 (n groups):
# 同步: ΣS_j + ΣC_j + ΣR_j
# 重叠: max(S_0 + ΣS_j, C_0) + ... + max(R_{n-1}, C_{n-1})
# 优化: 将最快的 S_{i,0}(group内通信) 和 R_{i,n-1}(ring通信) 放首尾
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 中基于 FastMoE 的 CUDA stream 基础设施实现，每个 worker 创建独立的 comm stream 和 comp stream。调度逻辑按 step j 展开 S/C/R 操作序列。在 *johnny* 和 *trevor* 上实测：智能调度单独加速 1.40×，与 dynamic shadowing 联合加速 2.20×（johnny）/ 5.72×（trevor）。理论上界为 (Lat_comm + Lat_comp) / max{Lat_comm, Lat_comp}，大模型和更多 worker 下实际加速比更接近理论上界（因启动开销相对更低）。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

FlowMoE 通过 Unified Pipeline Scheduling 将重叠范围从 MoE 层内扩展到整个 Transformer block——MHA 计算与 A2A 通信重叠（Pipe-AT 贡献 +10.3%），all-reduce chunk 与 A2A 通信间隙重叠（Pipe-AR 贡献 +24.6%），使计算-通信重叠率达到全 block 级别。
