## Multi-Tile Kernel with Runtime Tile Selector（多Tile内核与运行时Tile选择器）

术语是什么？通过联网搜索让回答具体和精准。
Multi-Tile Kernel是PAT提出的GPU attention kernel设计方法：不再使用单一硬编码的tile size配置（如FlashAttention固定m=64,n=128或FlashInfer固定m=16,n=128），而是offline求解一组可行的(m,n) tile配置，并为每个CTA在运行时选择最优配置。Tile selector是配套的运行时决策逻辑：对每个CTA，根据其query数q选择最小可行Q tile size m（round-up规则，如q=20→m=32避免64的padding浪费），根据其KV length选择最优KV tile size n（长KV偏大n降低per-SM concurrency减少tail execution bubble，短KV偏小n避免最后tile的compute bubble）。该设计解决了已有KV-centric kernel的one-size-fits-all资源浪费问题（当共享prefix的query数少于固定m时需要padding浪费shared memory/register，当CTA KV长度差异大时固定n造成execution bubble）。Ablation显示，替换为固定tile的PAT-fixed比完整PAT慢39%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multi-tile kernel的设计分为offline和online两个阶段：

```
// ============ Offline: Feasible Tile Set Derivation ============
Input: GPU hardware params (S_smem, S_reg_thr, S_register, S_num, L, B, h, b, b')
Output: feasible_tile_set = {(m,n)} 满足所有约束

For m in {16, 32, 64, 128}:   // power-of-2, >=16 (CUTLASS requirement)
  For n in {16, 32, 64, 128, 256}:
    // Constraint 1: Shared memory upper bound
    if m*h*b + n*h*b + m*h*b' > S_smem:  continue
    
    // Constraint 1: Register bounds (via offline compilation)
    R_thr, R_CTA = compile_and_profile(m, n)
    if R_thr > S_reg_thr:  continue
    CTA_per_SM = floor(S_register / R_CTA)
    if CTA_per_SM < 1:  continue
    
    // Constraint 2: Bandwidth lower bound
    D_flight = S_num * CTA_per_SM * n * h * b
    if D_flight < L * B:  continue  // insufficient in-flight data
    
    // Constraint 3: CUTLASS requirement (implicitly satisfied by loop)
    feasible_tile_set.add((m, n))

// A100 result: 11 feasible configs
// H100 result: 12 feasible configs (移除64,32和64,64)
```

```
// ============ Online: Runtime Tile Selection Per CTA ============
Input: CTA with q queries, KV length kv_len
Output: (m, n) tile configuration

// Q tile selection: round-up rule
m = min{mi in feasible_m_set | mi >= q}
// e.g., q=1→m=16, q=20→m=32, q=40→m=64, q=100→m=128

// KV tile selection: piecewise decision tree (offline profiled)
n = DecisionTree(kv_len):
  if kv_len <= 64:    return 16
  elif kv_len <= 256: return 32
  elif kv_len <= 1024: return 64
  elif kv_len <= 4096: return 128
  else:               return 256  // 需配合Long-KV Split
```

Kernel equivalence验证：在无共享prefix和execution bubble的batch下（batch size设为所有配置CTA concurrency的公倍数，A100用1134，H100用1188），所有feasible配置达到83%-86%（A100）或92%-94%（H100）带宽利用率且latency差异<2%，证明了tile selector可在不损失单kernel性能的前提下实现自适应。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Multi-tile kernel以CUTLASS/CuTe实现，每个feasible (m,n)配置编译为一个独立的kernel variant。Offline solver需要在目标GPU上运行micro-benchmark获取：memory latency L和bandwidth B（通过不同data size的global→shared transfer测latency vs data size曲线），per-thread和per-CTA register使用量（通过nvcc编译+static analysis获取）。移植到新GPU架构需重新运行offline solver推导等价tile set。PAT在A100和H100上都验证了该方法的通用性。Multi-tile kernel使用方式和PAT整体一致：通过pybind11暴露为vLLM backend，环境变量启用。在典型batch中，每个decode step使用的active tile config数量为1-5个（共11个feasible configs中）。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

