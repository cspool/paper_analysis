## Grouped Pairwise Exchange

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped Pairwise Exchange 是 FasterMoE（PPoPP'22）提出的细粒度 all-to-all 通信调度算法，将粗粒度的同步 all-to-all 操作拆分为 n 个 group 的逐 stride pairwise exchange 序列。n 个 group 形成环结构，在第 j 步（j=0,1,...,n-1），group i 向 group (i+j) mod n 发送数据并从 group (i-j) mod n 接收数据（stride 递增）。Group 分配采用启发式：将拓扑邻近的 workers 放入同一 group，使得 group 内（stride=0）通信最快。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Grouped Pairwise Exchange in one MoE layer (forward)
# n groups of workers, arranged in a ring
# Comm stream 和 Comp stream 独立并行

# Comm stream:
for j in 0..n-1:
    S_{i,j}:  send tokens to group (i+j) mod n
              recv tokens from group (i-j) mod n

for j in 0..n-1:
    R_{i,j}:  recv expert outputs from group (i+j) mod n
              send local token outputs to group (i-j) mod n

# Comp stream (与 Comm stream 并行执行):
for j in 0..n-1:
    C_{i,j}:  compute on tokens from group (i-j) mod n using local experts

# 依赖关系:
# C_{i,j} 依赖 S_{i,j} 完成 (token 已到达)
# R_{i,j} 依赖 C_{i,j} 完成 (计算结果可用)
# S_{i,j} 完成前不能启动 C_{i,j}

# 智能调度: 最快操作放在首尾
# S_{i,0}: group内通信, 无上层连接 → 最快 → 第一位
# R_{i,n-1}: ring通信, 全带宽利用 → 第二位快 → 末位
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 中基于 FastMoE 扩展实现。Group 大小通过环境变量 `FMOE_FUSE_GRAN` 控制。NCCL 用于底层 pairwise 通信。在 *johnny* 和 *trevor* 集群上评估，智能调度单独加速 1.40×，与影子化联合加速 2.20×（johnny）/ 5.72×（trevor）。理论加速上界为 `(Lat_comm + Lat_comp) / max{Lat_comm, Lat_comp}`，某些层达到理论上界的 99%。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
