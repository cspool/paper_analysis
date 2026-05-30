## Device-Limited Routing (设备限制路由)

术语解释
Device-Limited Routing 是 DeepSeek-V2 提出的一种 MoE 路由约束机制，在细粒度专家分割（大量小专家）场景下，限制每个 token 的目标专家最多分布到 M 个设备上，从而控制 expert parallelism 下的 all-to-all 通信开销。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
在 DeepSeek-V2 的 160 routed experts 配置中，采用 8-way expert parallelism (D=8)，每个设备部署 20 个 routed experts。若不做限制，top-K (K_r=6) 选择可能将 6 个专家分布在多达 6 个设备上，导致每 token 的 MoE 通信量翻倍。Device-Limited Routing 分两步：(1) 先按 token-to-expert affinity 选出 M 个最受青睐的"设备"（而非直接选 expert）；(2) 再在这 M 个设备的 expert 子集中做 top-K 选择。DeepSeek-V2 设置 M=3，实验表明 M≥3 时性能与无限制 top-K 路由接近对齐。

为什么需要？MoE 通信频率与目标专家覆盖的设备数成正比。DeepSeekMoE 的细粒度专家分割导致激活专家数多（K_r=6），若不加设备限制，expert parallel all-to-all 通信量会严重拖累训练效率。Device-Limited Routing 将通信量从 O(K_r) 限制到 O(M)，当 M=3 时通信量减少约 50%。

从算法pipeline角度拆解术语：
```
=== Device-Limited Routing (per token) ===

Input: u_t (token hidden state), {e_i} (expert centroids), D=8 devices, M=3, K_r=6

// Step 1: Compute token-to-expert affinity
for i in 1..160:
    s_{i,t} = Softmax_i(u_t^T · e_i)

// Step 2: Aggregate affinity per device (20 experts per device)
for device d in 1..8:
    S_d = max_{i in experts(d)} s_{i,t}     // or sum/top-k aggregation

// Step 3: Select top-M devices
Devices_selected = TopK_devices({S_d | d=1..8}, M=3)

// Step 4: Top-K among experts on selected devices only
Experts_selected = TopK({s_{i,t} | expert i is on device d ∈ Devices_selected}, K_r=6)

// Step 5: Compute gating and FFN output (standard)
g_{i,t} = s_{i,t} if i in Experts_selected else 0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 HAI-LLM 训练框架中实现，与 expert parallel all-to-all 通信层紧密集成。训练时：每 token 最多与 M=3 个设备通信（而非 naive 的 K_r=6 个）。DeepSeek-V2-Lite 因为所有 expert 部署在同一设备上（无 expert parallelism），不需要 Device-Limited Routing。DeepSeek-V3 继用此设计（M=4, D=8, K_r=8）。

涉及论文标题：
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
