## KV Cache Sharing in MoE Speculative Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Sharing in MoE Speculative Decoding 是 MoE-SpeQ 的核心内存优化：draft (INT4) 和 target (FP16) 共享同一 KV cache。target verify 后将高精度（FP16）KV 写回 shared buffer；draft 在此高精度 KV 上推理，提供更准确上下文。效果：(1) VRAM 节省 43%（Qwen1.5-MoE 12K: 13.40→7.68GB）；(2) acceptance rate >90%（高于 Eagle 的 ~80%），因 draft 基于 target 的高精度 KV 而非独立维护的低精度 KV。

从系统架构角度拆解术语：
```
# 每 decode cycle
draft_output = draft_model.forward(kv_cache=shared_kv)  # 读 target 高精度 KV
target_output = target_model.forward(kv_cache=shared_kv)
shared_kv = target_model.kv_cache  # target 高精度 KV 覆写，draft KV 丢弃
```
需共享相同 hidden_dim/num_heads，draft 的 KV write 与 verify 阶段互斥。与 Parameter Sharing（共享 non-expert weights/embeddings/norms）协同实现完整 VRAM 节省。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
仅保留 target KV（FP16 高精度），draft INT4 推理产生的 KV 在 verify 后丢弃。论文未开源。

涉及论文标题：
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts
