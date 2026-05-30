## Llama.cpp (as MoE Inference Base Framework)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Llama.cpp (https://github.com/ggerganov/llama.cpp) 是一个用纯 C/C++ 编写的高效 LLM 推理框架，支持 CPU 和 GPU 混合计算，广泛应用于边缘设备和消费级硬件上的 LLM 部署。其核心设计：将足够多的完整 Transformer 层置于 GPU memory，剩余层存储于 CPU memory 或 SSD；GPU 层处理完成后将中间激活传递到 CPU，由 CPU 完成剩余层计算。对 MoE 模型，Llama.cpp 原生支持有限（按 per-layer 而非 per-expert 粒度管理权重），需要修改权重分布和计算模式才能高效利用 MoE 的稀疏激活特性。HOBBIT 和 MoE-APEX 等系统在 Llama.cpp 上构建，增加 8000+ 行 C++/C 代码实现 per-expert 多精度 offloading。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Llama.cpp 的原生推理流程（以 dense 模型为例）：

```
# 初始化
1. 加载 model.gguf → 解析权重、tokenizer、超参数
2. 分配 GPU memory：将前 N 层完整权重从 CPU/SSD 加载到 VRAM
3. 标准 compute graph：Embedding → [GPU layers] → CPU layers → LM head

# Per-token 执行
输入: token_id
1. embedding[token_id] → hidden state x (CPU)
2. 将 x 上传到 GPU (cudaMemcpy H2D)
3. for layer in GPU_layers:
     x = Attention(layer, x)   # Flash Attention or standard
     x = FFN(layer, x)         # Standard MLP
4. 将 x 下载到 CPU (cudaMemcpy D2H)
5. for layer in CPU_layers:
     x = Attention_CPU(layer, x)  # 多线程 CPU 计算
     x = FFN_CPU(layer, x)
6. logits = LM_head(x)  # CPU
7. next_token = sample(logits)
```

HOBBIT/MoE-APEX 对 Llama.cpp 的修改（MoE 支持）：

```
# 修改的权重分布
GPU VRAM:
  - 所有 non-expert 权重 (Attention + LayerNorm + Embedding)
  - expert cache (High-Precision Cache + Low-Precision Cache)
CPU Memory / SSD:
  - 所有 expert 权重 (每个 expert 的 W_g, W_p, W_o)，每 expert 多精度版本

# 修改的 MoE layer 计算
for layer in range(L):
    x = Attention(layer, x)
    # === MoE 模块 (替代原 FFN) ===
    gate_out = Gating(W_gate[layer], x)  # GPU
    topk_ids = topk(gate_out, K=K)
    
    # 异步预取：Stacking Computer 预测后续层 expert
    pred_experts = StackingComputer(x, W_gate[layer+1:])
    
    # Expert 加载：Dynamic Expert Loader
    for e in topk_ids:
        if e not in expert_cache:
            score = compute_unimportance_score(e, gate_weights)
            precision = "high" if score <= T1 else "low" if score <= T2 else "skip"
            async_load_expert(e, precision)  # read() from CPU/SSD
    
    # Expert 计算
    y = sum(gate_weights[e] * ExpertFFN(x, e) for e in topk_ids if not skip)
    x = x + y  # residual
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **模型格式**：LLM 权重以 GGUF 格式存储（量化友好的自定义二进制格式），HOBBIT 为每个 expert 额外存储 INT4/INT2 量化版本
- **后端支持**：CUDA (NVIDIA GPU)、Metal (Apple Silicon)、Vulkan (跨平台)、CPU (AVX2/AVX-512/NEON)
- **内存管理**：mmap() 将模型文件映射到虚拟地址空间，操作系统按需换页。对 MoE，这会导致严重 page fault（因需随机访问不同 expert），HOBBIT 改用显式 read() + pinned memory 避免此问题
- **集成方式**：HOBBIT 作为 fork/patch 形式提供，修改 `llama.cpp` 核心库的 `llama_model_loader` 和 `llama_eval_internal` 函数，新增 `expert_loader`、`expert_predictor`、`cache_manager` 三个模块
- **开源状态**：Llama.cpp 本身 MIT 协议开源；HOBBIT 代码未公开完整仓库

涉及论文标题：
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading
