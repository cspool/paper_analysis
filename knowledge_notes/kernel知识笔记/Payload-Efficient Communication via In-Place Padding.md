## Payload-Efficient Communication via In-Place Padding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Payload-Efficient Communication 是 FlashMoE 提出的通信优化：消除 MoE token dispatch 中因 expert capacity padding 导致的冗余网络传输。传统 MoE (DeepSpeed) 中 AlltoAll 的 buffer 大小必须预先约定 (C×H)，当某 expert 接收 token 少于 C 时，不足 slot 零填充——这些零值占用通信带宽并可能触发无意义计算。FlashMoE 改为在本地 symmetric tensor buffer 内完成 padding (in-place padding)，仅将包含实际 token 的 tile 通过网络传输。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

对比: DeepSpeed sendbuf[j] 预分配 C×H×sizeof(fp16) = 1MB，实际 100 tokens → 61% null payload。FlashMoE: actual_tiles = ceil(actual_tokens/128), 每 tile 本地补齐后仅传输实际 tile → nvshmem_putmem(remote_L, tile_data, TILE_SIZE)。通信量 ∝ ceil(actual_tokens/128)×H 而非 ceil(C/128)×H。expert 分布不均匀时收益最大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现: (1) C 对齐到 bM=128 倍数；(2) 本地 padding: token 写入 L OUTGOING slot 前补零至 128 rows (纯 local write)；(3) Network 仅传输含有效 token 的 tile。Size(L) worst-case (S/E < bM): 4×(bM×E/S)×Size(T)。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
