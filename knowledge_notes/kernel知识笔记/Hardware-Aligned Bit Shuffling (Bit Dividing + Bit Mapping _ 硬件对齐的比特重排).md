## Hardware-Aligned Bit Shuffling (Bit Dividing + Bit Mapping / 硬件对齐的比特重排)

术语是什么？通过联网搜索让回答具体和精准。
Hardware-aligned bit shuffling是Quantix提出的一种离线权重布局转换技术，用于解决3-bit non-uniform quantized weights与GPU native word size（32/64-bit）及Tensor Core operand layout之间的不对齐问题。核心idea是将"odd-bit packing"难题转换为"easy-bit packing"问题：通过bit dividing将每个3-bit index拆分为1-bit和2-bit两个独立矩阵，使每路均与GPU INT类型天然对齐；再通过bit mapping将元素按Tensor Core tile访问模式重排为连续segments，确保coalesced memory access和Tensor Core operand匹配。该变换是lossless w.r.t. quantized model——只重排index bits，不修改centroids，因此完全保留原non-uniform量化精度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bit shuffling分两步，均在offline完成：

```
// ===== Step 1: Bit Dividing =====
// 输入: Wq ∈ INT^{K×M} (每个元素3-bit)
// 输出: Wq,1 ∈ {0,1}^{K×M}, Wq,2 ∈ {0,1,2,3}^{K×M}

for each element e in Wq:
    // 3-bit binary: e = b2 b1 b0 (MSB to LSB)
    // 任意选1个bit分离，例如MSB:
    Wq,1[i,j] = (e >> 2) & 0x1   // 1-bit (可以选任意bit)
    Wq,2[i,j] = e & 0x3          // 2-bit (剩余)

// Packing: 32个1-bit元素 → 1个32-bit word (perfect fit)
//          32个2-bit元素 → 1个64-bit word (perfect fit)
// 关键: 1和2都是32和64的因子，因此无padding浪费、无spanning跨界
```

```
// ===== Step 2: Bit Mapping =====
// 目标: 将packed elements按Tensor Core (TC) tile访问模式重排
// 每个warp负责1个64×64 warp tile
// warp tile分为16个16×16 TC tiles (TC Tile 0-15)

// 在每个TC tile内，每个thread负责4 pairs of elements
// Pair 0: element (0,0/1), element (8,0/1)
// Pair 1: element (0,8/9), element (8,8/9)
// Pair 2: element (0,0/1), element (8,0/1) [不同位置]
// Pair 3: element (0,8/9), element (8,8/9) [不同位置]

// 按配对的元素索引，批量映射输出

// Bit mapping: 收集同一thread在16个TC tiles中的4×16 pairs×n bits
// = 128n bits (n=1: 128 bits; n=2: 256 bits)
// 组织为连续linear memory segment Wn'

// 每个thread的segment布局:
// W1': 128 bits → 1次cp.async (128-bit)抓取
// W2': 256 bits → 2次cp.async (128-bit)抓取
```

```
// ===== Physical layout in memory =====
// warp 0: [thread0_W1' | thread1_W1' | ... | thread31_W1']
//         [thread0_W2' | thread1_W2' | ... | thread31_W2']
// warp 1: [thread0_W1' | thread1_W1' | ... | thread31_W1']
//         ...
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit shuffling实现为Python/C++预处理脚本，在模型部署前执行一次。输入为已non-uniform quantized的权重（由SqueezeLLM/Any-Precision/Bitsandbytes等生成），输出为W1'、W2'和reordered centroids C。具体使用：
1. 加载由任意non-uniform量化方案生成的Wq (3-bit indices)和C (per-row FP16 centroids)
2. 执行bit dividing: Wq→Wq,1 (1-bit) + Wq,2 (2-bit)
3. 执行bit mapping: 按warp tile→TC tile层次重排，生成W1'/W2' linear segments
4. 输出打包为GPU可读取的binary格式
该变换的代价被所有后续推理分摊（offline, one-time），因此对在线推理无性能影响。Quantix论文开源：https://github.com/yuang-chen/Quantix-PPoPP26

涉及论文标题：
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

