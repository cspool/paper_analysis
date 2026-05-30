## EP+TP Hybrid Parallelism for MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

EP+TP Hybrid Parallelism 是 IFMoE 提出的针对 fine-grained MoE 推理的混合并行策略。核心思想：在传统 Expert Parallelism (EP) 的基础上，对共享参数（Attention 权重、LayerNorm、Shared Expert）使用 Tensor Parallelism (TP) 切分，而非 EP 的每卡全量复制。Expert 参数保持 EP 分布不变。

设计动机：(1) 传统 EP 推理时每台 machine 复制完整的共享参数，多卡时大量显存被冗余的 Attention/Norm 参数浪费；(2) 推理场景下通信通常在节点内（NVLink/PCIe），带宽充足，TP 引入的 AllReduce 通信开销可接受；(3) Fine-grained MoE 的单个 expert 尺寸较小，EP 的 expert 负载均衡通常良好。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

IFMoE EP+TP Hybrid 推理流程（以 Qwen2 在 4× A6000 上为例）：

```
# 4 GPUs, EP+TP Hybrid:
# - Attention/Norm/Shared Expert params: TP 切分到 4 卡 (每卡 1/4)
# - Routed Expert params: EP 分布 (每卡持有 64/4=16 experts)
# - 所有 GPUs 处理相同的输入 tokens (Same Tokens, ST)

For each layer:
    # 1. Attention (TP): 每卡持有 1/4 Attention 权重
    GPU_i: Q_i,K_i,V_i = W_Q_i @ X, W_K_i @ X, W_V_i @ X
    # AllReduce 收集完整 attention output
    
    # 2. Router (local): 每卡独立计算 routing
    gate_scores = Router(norm_hidden)        # [B, 64]
    topk_indices = TopK(gate_scores, k=6)    # 选 6 experts
    
    # 3. Expert Computation (EP): double All-Gather
    # All-Gather 1: 收集各 GPU 的 routed tokens per expert
    # All-Gather 2: 收集各 GPU 的 expert outputs
    # 替代传统 All-to-All dispatch/combine
    
    # 4. Shared Expert (TP): 同 Attention，TP 切分
    shared_out_i = SharedExpert_i(norm_hidden)
    shared_out = AllReduce(shared_out_i)
    
    output = attn_out + routed_out + shared_out
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

IFMoE 的 EP+TP Hybrid 内存节省：
- Deepseek-Lite (64 experts, 2 machines): 节省 4.6GB per machine
- Qwen2-57B-A14B (64 experts, 4 machines): 节省 10GB per machine
- Deepseek-v2 (160 experts, 8 machines): 节省 23GB per machine

通信变更：传统 EP 使用 All-to-All dispatch + combine，IFMoE 改用 double All-Gather。因为在节点内推理场景下 All-Gather 的通信量与 All-to-All 相当（令牌数量相同，每令牌传输 d_model 维数据），但配合 EP+TP 混合并行后内存效率更高。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
