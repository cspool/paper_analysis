## ASTRA-SIM (Distributed DL Communication Simulator)

术语解释
ASTRA-SIM 是 Georgia Tech 开发的分布式深度学习通信模拟器，用于模拟大规模 GPU/加速器集群中的集合通信（AlltoAll、AllReduce、AllGather 等）在不同网络拓扑、带宽和并行配置下的延迟和带宽性能。DualSparse-MoE 使用 ASTRA-SIM 模拟 ETP vs S-ETP 在 NVL72 和 CloudMatrix384 配置下的通信带宽。

术语是什么？
ASTRA-SIM 的输入：workload communication pattern (collective types, message sizes) + topology specification (节点数、GPU 数、intra/inter-node bandwidth、NVLink/PClink/IB 配置、collective algorithm)。通过模拟 collective 在底层拓扑上的执行过程，输出通信延迟、带宽和 bottleneck 分析。DualSparse-MoE 在中使用 ASTRA-SIM 在 NVL72 (EP=9, TP=8) 和 CloudMatrix384 (EP=48, TP=8) 配置下比较 ETP vs S-ETP 通信带宽：S-ETP 实现 10.2-80.4% (NVL72) 和 9.9-28.3% (CM384) improvement。

从硬件架构角度拆解术语：
```
=== ASTRA-SIM Simulation Pipeline ===
Input: topology.json + workload.yaml
  Topology: GPU count, NVLink bandwidth per link, latency, switch hierarchy
  Workload: collective list [(type, size, participants), ...]

Simulate per collective:
  Select algorithm (ring/tree/hierarchical) based on topology+size
  Compute per-hop delay = chunk_size / link_bandwidth
  Aggregate total_comm_time = max(critical_path)
  Compute achieved_bandwidth = total_data / total_comm_time

DualSparse-MoE usage:
  Pattern A (ETP):  AlltoAll + AllGather + ReduceScatter + AlltoAll
  Pattern B (S-ETP): AlltoAll + AlltoAll
  Result: S-ETP bandwidth ↑ 10.2-80.4% (NVL72)
```

术语一般如何实现？如何使用？
- 开源：https://github.com/astra-sim/astra-sim (Georgia Tech ASTRA Lab)
- 配置：topology JSON (GPU 数, link BW/latency, switch topology) + workload YAML (collectives)
- 支持：hierarchical all-to-all、NCCL algorithm modeling、custom topology injection
- 局限：仅模拟通信不建模计算 → improvement 为 upper bound；topology 需手动指定
- 论文中用途：补充 real-world 8×H20 小实验局限，验证大规模全互联系统 S-ETP 通信优势

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
