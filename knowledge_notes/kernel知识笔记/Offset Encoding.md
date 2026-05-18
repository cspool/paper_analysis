## Offset Encoding

术语是什么？通过联网搜索让回答具体和精准。
Offset Encoding是Focus SEC中的一个紧凑位置编码方案，用于在semantic pruning后记录保留token的相对空间位置。由于SEC pruning破坏了token的连续空间结构（移除不重要的tokens），下游SIC需要知道每个retained token的原始(Frame, Height, Width)坐标以构建2×2×2时空block。Offset Encoder用sliding window为每对连续保留token记录一个小整数offset（相对于前一保留token的position delta），而非存储绝对坐标。这种compact encoding仅需lightweight registers，无需global memory access。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Offset Encoding的生成和使用：
```
# Generation (in SEC, after top-k selection):
# retained_indices: sorted indices of top-k tokens in original sequence
# Original token positions are in FHW order: (f, r, c) → linear index

prev_pos = retained_indices[0]
offset_stream = [prev_pos]  # first entry: absolute position of first retained token
for idx in retained_indices[1:]:
    delta = idx - prev_pos  # offset to previous retained token
    offset_stream.append(delta)
    prev_pos = idx

# offset_stream stored as compact ints, streamed alongside GEMM output

# Usage (in SIC layouter):
# Reconstruct absolute positions from offset stream:
positions = [offset_stream[0]]
for delta in offset_stream[1:]:
    positions.append(positions[-1] + delta)

# Map linear position to FHW coordinates:
for pos in positions:
    f = pos // (H * W)      # frame index
    r = (pos % (H * W)) // W  # row index
    c = pos % W               # column index
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Focus SEC中用lightweight registers实现：sliding window over retained token indices → compute delta → output as compact int。完全local和streaming，与SEC sorter同步输出。开源实现见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

---

