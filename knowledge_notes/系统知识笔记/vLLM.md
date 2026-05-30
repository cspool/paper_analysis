## vLLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
vLLM 是 UC Berkeley 开发的高吞吐量 LLM 推理和服务引擎（Kwon et al., SOSP 2023），核心创新是 PagedAttention——将 KV Cache 管理建模为操作系统中的虚拟内存分页问题，通过 block-level 的 KV Cache 分配和共享实现接近零的显存碎片，从而在相同硬件上支持更大的 batch size 和更高的吞吐量。关键特性：(1) Continuous Batching —— 动态地将新到达的请求加入当前 batch，无需等待整个 batch 完成；(2) PagedAttention —— KV Cache 以固定大小的 block 为单位分配，可动态增长且支持 prefix sharing（相同 prompt prefix 的请求共享同一份 KV Cache blocks）；(3) 高吞吐量——相比 HuggingFace Transformers 实现 24x 吞吐量提升。支持多种模型架构（LLaMA, Qwen, InternLM 等）和量化方法（GPTQ, AWQ, FP8 等）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
vLLM 的请求处理流程：
```
# vLLM Scheduler 主循环
while True:
    # 1. 从 waiting queue 取请求
    new_requests = dequeue(waiting_queue)
    
    # 2. 为每个请求分配 KV Cache blocks
    for req in new_requests:
        req.blocks = allocate_blocks(req.max_tokens, block_size=16)
        running_batch.append(req)
    
    # 3. 模型 forward (continuous batching)
    for req in running_batch:
        if req.is_prefill:
            # Prefill: 一次性处理所有 prompt tokens
            # PagedAttention: KV cache 写入预先分配的 blocks
            kv_blocks = req.blocks
        else:
            # Decode: 逐 token 生成
            # PagedAttention: 读取已存在的 KV blocks + 写入新 token
    
    # 4. 完成检查
    for req in running_batch:
        if req.done():
            free_blocks(req.blocks)  # 释放 KV Cache blocks
            running_batch.remove(req)
```

在 DIG 中的使用：vLLM 被用于加速 query identification（LLM 推理）和 reward assignment（LMM 推理）两个阶段，利用其 continuous batching 和 PagedAttention 提高多帧评分的吞吐量，将大量候选帧的逐个评分高效地批量处理。论文报告的总推理时间（Table 9-10）均基于 vLLM 后端。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源地址：https://github.com/vllm-project/vllm。安装：`pip install vllm`。使用：`python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-7B-Instruct`。DIG 在该框架中作为 LMM 推理的后端，不需要修改 vLLM 源码——直接通过其 OpenAI-compatible API 或 Python API 调用。关键配置参数：max_num_batched_tokens（控制 batch 的最大 token 数）、max_num_seqs（最大并发序列数）、gpu_memory_utilization（GPU 显存利用率阈值）。在 DIG 的 reward assignment 中，vLLM 可以同时处理多个 r-frame 的评分请求，大幅减少逐个调用 LMM 的排队延迟。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
- SAGE__Training_Smart_Any-Horizon_Agents_for_Long_Video_Reasoning_with_Reinforcement_Learning

SAGE 中的 vLLM 使用方式：
SAGE 使用 vLLM 作为所有模型（包括 SAGE-MM orchestrator 和所有 DIRECT/AGENT baselines）的推理评估引擎。温度设为 0.0，对因 JSON 格式不符导致的非确定性输出以 temperature=0.7 重试最多 4 次。在 inference runtime 比较（Table 11）中，SAGE 的 8.6s/sample 远快于其他 AGENT baselines（VideoMind 24.7s, LVAgent 92.9s, VideoExplorer 137.7s, VideoAgent 1445.0s），其中 vLLM 的 continuous batching 和 PagedAttention 是保持低延迟的关键。
