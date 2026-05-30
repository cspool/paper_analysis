## CPU vs GPU Batching Effects in MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CPU vs GPU Batching Effects 是 Fiddler 论文通过 microbenchmark 揭示的关键性能特征差异，是其动态执行策略决策的基础。核心发现：
- **GPU 端**：expert FFN 执行延迟近乎恒定，与输入 batch size s 无关（s=1..64 范围内延迟变化 <10%）。这是因为 GPU 的并行计算能力使执行延迟受限于从显存加载参数的时间（memory-bandwidth bound），而非计算时间。
- **CPU 端**：expert FFN 执行延迟随输入 batch size s 近乎线性增长。这是因为 CPU 的计算能力远弱于 GPU，延迟受限于计算（compute-bound），参数加载时间被计算时间完全掩盖。
- **PCIe 传输**：weight copy (CPU→GPU, ~300MB/expert) 延迟恒定，是 GPU computation 的 2-5×；activation copy (GPU→CPU, s×4096×2 bytes) 延迟极小，<1% of single-input CPU latency。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fiddler microbenchmark 的测量方法和建模结果（Appendix A, Figure 7）：

```
// Microbenchmark 测量 (32 layers × 多次, 报告 mean ± std):
// 
// GPU execution latency vs input size s:
//   GPU(s=1):  基准时间
//   GPU(s=2):  ~1.0× 基准
//   GPU(s=4):  ~1.0× 基准
//   GPU(s=8):  ~1.0× 基准
//   GPU(s=16): ~1.0× 基准
//   GPU(s=32): ~1.0× 基准
//   GPU(s=64): ~1.0× 基准
//   → Model: gpu_lat(s) = gpu_const
//   (Env1 s=1 因 PyTorch 单 batch 不同实现有 ~10% 差异，可忽略)
//
// CPU execution latency vs input size s:
//   CPU(s=1):  基准时间
//   CPU(s=2):  ~2.0× 基准
//   CPU(s=4):  ~4.0× 基准
//   CPU(s=8):  ~8.0× 基准
//   ...
//   → Model: cpu_lat(s) = cpu_slope × s
//
// PCIe transfer latency:
//   Weight copy (CPU→GPU): constant, 2-5× GPU computation
//   → Model: trans_lat() = trans_const
//
//   Activation copy (GPU→CPU): <1% of CPU(s=1)
//   → Model: ignored in latency model

// Algorithm 1 中的决策函数:
gpu_lat(s) = gpu_const                   // 恒定
cpu_lat(s) = cpu_slope × s              // 线性
trans_lat() = trans_const               // 恒定

// 决策阈值:
// cpu_lat(s) < gpu_lat(s) + trans_lat()
// → cpu_slope × s < gpu_const + trans_const
// → s < (gpu_const + trans_const) / cpu_slope = s_threshold
//
// s < s_threshold → Strategy (c): CPU 执行
// s ≥ s_threshold → Strategy (b): GPU+transfer 执行
```

延迟构成的 breakdown（以 Mixtral-8x7B expert 为例）：
| Component | Latency | Bound | s-dependence |
|-----------|---------|-------|-------------|
| GPU expert FFN | ~T_gpu | Memory BW | Constant |
| CPU expert FFN | ~s × T_cpu_per_token | Compute | Linear in s |
| PCIe weight copy | ~T_wcopy (300MB) | PCIe BW | Constant |
| PCIe activation copy | ~s × T_acopy (negligible) | PCIe BW | Linear in s (但可忽略) |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **Profiling 方法**：初始化阶段对每个 operation 运行 32 次（每层 1 次），记录 mean ± std，一次性完成
- **适用条件**：batching effects 的差异来源于 GPU 和 CPU 的架构根本差异——GPU 的 SIMT 大规模并行 vs CPU 的少量大核心——因此该特征在各类 GPU/CPU 组合中普遍成立
- **对调度的影响**：正是这种 "GPU 恒定 vs CPU 线性" 的差异使得动态策略选择有意义——若两者都是线性的或都是恒定的，则总有单一最优方案
- **Fiddler 利用方式**：在 initialization 阶段测量三个常数（gpu_const, cpu_slope, trans_const），runtime 仅需查询 s 并比较

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
