## MM-SP Inference Mode (Multi-Modal Sequence Parallelism Inference)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MM-SP推理模式是LongVILA提出的基于序列并行的VLM分布式推理系统，解决标准HuggingFace Pipeline Parallelism推理的两个核心问题：(1) 计算效率低——HF Pipeline逐层分配GPU，同一时刻仅1个GPU活跃计算，其余idle；(2) 内存分布不均——HF Pipeline首卡需存储全部输入embedding和图像数据（96K序列下首卡80GB activations，其余卡仅18GB），内存瓶颈由首卡决定。MM-SP推理通过序列并行所有GPU并发参与每个transformer layer的计算（accelerating proportional to GPU count），同时将序列切分到所有GPU，内存均匀分布。实测8 GPU下实现8.2×加速和2.9×更长序列。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# MM-SP Inference vs HF Pipeline Inference (4 GPUs, 256K tokens)

# HF Pipeline Parallelism (4 stages):
#   GPU0: Embedding + Layers 0-7   → passes activations to GPU1
#         Stores: all input embeddings + images (bottleneck)
#   GPU1: Layers 8-15              → waits for GPU0, then passes to GPU2
#   GPU2: Layers 16-23             → waits for GPU1, then passes to GPU3
#   GPU3: Layers 24-31 + LM Head   → waits for GPU2
#   Timeline: |GPU0_active| GPU1_idle |GPU1_active| GPU2_idle |GPU2_active| ...
#   Only 1 GPU active at any time
#   Memory: GPU0 stores ALL embeddings [256K, 4096] ≈ 4GB embeddings alone

# MM-SP Inference (sp=4, 4×1 mesh):
#   Stage1 Sharding:
#   GPU0-3 each encode N_frames/4 images → balanced vision encoding
#   
#   Stage2 Sharding:
#   GPU0: tokens[0:64K], GPU1: tokens[64K:128K], 
#   GPU2: tokens[128K:192K], GPU3: tokens[192K:256K]
#   Each GPU holds 1/4 of embeddings → 4× less memory per GPU
#   
#   Per-Layer (all GPUs active simultaneously):
#     GPU0-3: QKV Projection (local)
#     GPU0-3: A2A(Q,K,V) reshuffle by head dim (intra-node)
#     GPU0-3: FlashAttention2(Q_local, K_all, V_all)
#     GPU0-3: Reverse A2A + FFN (local)
#   
#   Decoding Loop:
#     GPU that holds the last sequence position → generates next token
#     Broadcast new token to all GPUs
#     All GPUs update their KV cache in-place
#     Detect EOS → All-reduce termination signal

# Key differences from training mode:
# 1. KV cache management: incremental update each decode step
# 2. Position encoding: dynamic update as sequence grows
# 3. Termination: detect last-token holder → broadcast termination
# 4. No backward pass, no gradient/optimizer states
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于HuggingFace Transformers通过monkey-patching集成，调用方式与标准Transformers推理兼容。推理模式需额外处理：(1) 动态tensor——每个decode step的input tokens和position encodings变化；(2) KV cache——每个GPU管理本地KV cache的增量更新而非重计算；(3) 终止协调——持有最后token位置的GPU检测到EOS后通过all-reduce广播终止信号。适用于需要处理超长视频（数百到数千帧）的VLM推理场景，可通过增加GPU数量线性扩展最大序列长度。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
