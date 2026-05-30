## Multi-LoRA Dynamic Switch for MoE Serving

术语解释
Multi-LoRA Dynamic Switch 是 MELD（KDD '24）提出的一种基于 Punica + vLLM 的多 expert 推理服务方案。通过在单个 base LLM 上动态加载和切换多个 LoRA adapter（每个 adapter 对应一个 expert），实现在 consumer GPU 上的低延迟 MoE 推理，避免传统 MoE 模型需要的大显存多 GPU 部署。

术语是什么？
MELD 的推理系统将 MoE 的 "expert = 子网络参数" 映射为 "expert = LoRA adapter"。每个 expert 是一个独立训练的 LoRA 权重（约 10-50MB），而非完整模型的参数子集。三个关键优势：
1. **存储效率**：n 个 expert = n 个 LoRA adapter + 1 个 base model
2. **动态切换**：Punica 支持单 GPU 同时 serving 1 base model + 最多 200 LoRA weights
3. **无 Merge 开销**：避免 LoRA merge 到 base model，仅 concatenation + forward pass 按需应用。Model process time 比 JellyFish 快 10×，比 Mixtral 快 30×

从系统架构角度拆解术语。
Serving 流程（query → GPU 输出）：
```
1. Query 到达 → Serializer 统一 dict 格式
2. M_RAG 编码 → emb_q
3. Router N(emb_q) → top-k expert IDs (k=3)
4. Punica Scheduler:
   - 检查 expert LoRA 是否在 GPU VRAM
   - 不在则从 CPU RAM 加载 (~10-50MB, <1ms)
   - 相似 query 聚合到同 GPU (load balancing)
5. vLLM 执行:
   - Prefill: batch tokens → Base Mistral-7B forward
   - 激活 k 个 LoRA adapter，各 expert 独立推理
   - 加权融合: y = Σ g_i * (W_0·x + B_i·A_i·x)
   - Decode: PagedAttention KV cache, autoregressive 生成
6. 返回结果
```

关键性能：4×3090 吞吐 = 3.7× JellyFish(13B), 5.6× Mixtral(56B)；1×3090 MELD full precision 运行，Mixtral 无法部署（OOM）。

术语一般如何实现？如何使用？
- 基于 Punica + vLLM: PagedAttention + continuous batching + multi-LoRA
- Expert 训练: LLaMA-Factory + LoRA PEFT
- Base model: Mistral-7B (14GB FP16), 每 expert LoRA ~10-50MB
- 单 3090 (24GB): base + 200× LoRA = 14GB + 2-10GB
- 开源: https://github.com/authurlord/MELD

涉及论文标题：
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing
