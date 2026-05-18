## Four-Stage Radix Select TopK

术语是什么？
Four-Stage Radix Select TopK 是 VAR-Turbo 论文中 Radix Sort Core 采用的大 K TopK 选择 dataflow。将 TopK 问题从通用排序方案（如 Bitonic Sort + Merge Sort，在大 K 上需 O(N log² N) 比较和全局数据重排）转化为固定 4 阶段流水线，利用 radix 分解（按 digit 分桶）避免全局比较和重排。特别适用于 VAR 场景中大 K（如 N=4096 时 K=1936）confidence/importance selection。

从kernel调度角度拆解术语：
Four-Stage Radix Select 的计算过程（以 N=4096, K=1936, 8-bit radix 为例）：
```
// Stage 1: CountBin
bin_counts[256] = {0}
for each element i in [0, N):
  digit = (confidence[i] >> (radix_shift * 8)) & 0xFF
  bin_counts[digit]++

// Stage 2: PrefixSum
prefix_sum[0] = 0
for b in [1, 256):
  prefix_sum[b] = prefix_sum[b-1] + bin_counts[b-1]

// Stage 3: SelectBin
// 从最高 radix digit 开始搜索
for b in [255, 0]:
  if prefix_sum[b] <= K < prefix_sum[b] + bin_counts[b]:
    candidate_bin = b
    K_offset = K - prefix_sum[b]
    break

// Stage 4: Filter
candidates = []
for each element i in [0, N):
  digit = extract_radix_digit(confidence[i])
  if digit == candidate_bin:
    candidates.append((i, confidence[i]))
sort(candidates by confidence descending)
selected_indices = candidates[0:K_offset]
```
多轮 radix（从最高 digit 到最低 digit）迭代后精确选出 TopK。Locality-aware Scheduling 扩展：维护 history table（mask map）标记已解码区域，PE 分组在不同空间区域并行执行各自的 Radix Select，优先处理靠近已解码 token 的高置信行/block。

术语一般如何实现？如何使用？
在 VAR-Turbo accelerator 的 Radix Sort Core 中以 SystemVerilog RTL 实现（TSMC 28nm）。关键优势：相比 Bitonic Sort（O(N log² N) 比较器和 log² N 级流水线）和 Merge Sort（需全局 data shuffle），Radix Select 的 4 阶段流水线是固定深度的（与 K 和 N 无关），仅需 bin counter、prefix sum adder、comparator 和 filter mux。Locality-aware Scheduling 利用 2D 图像 token 的空间局部性进一步提升吞吐。该 dataflow 可推广至其他需在线大 K TopK 的场景（如 attention sparsification、KV cache pruning、MoE gating）。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

