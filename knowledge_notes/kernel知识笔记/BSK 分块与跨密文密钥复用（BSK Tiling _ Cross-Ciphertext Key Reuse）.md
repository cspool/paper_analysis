## BSK 分块与跨密文密钥复用（BSK Tiling / Cross-Ciphertext Key Reuse）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BSK（Bootstrapping Key）分块复用是 MNEMOS 针对 GPU 上 TFHE 外部乘积访存瓶颈的核心 kernel 级设计：一次 MAC 需取 (k+1) 倍于 GLWE 体积的 BSK（形状 (k+1)ℓ×(k+1)），且一批内多个 PBS 密文（同一卷积层共享参数）访问同一份 BSK。朴素做法（整 BSK 缓存进共享内存）不可行——部分参数集 BSK 超过 A100 每 SM 192KB 合并 L1/SPM 上限，且过度分配共享内存会蚕食 L1 缓存容量。由于 BSK 与傅里叶系数之间是逐元素 Hadamard 积（非一般矩阵乘），线程块无需持有整 BSK，只需处理对应的一块 TBSK 对一块 TGLWE；同一 BSK 分块被多个线程块并发复用（跨密文复用），把复用层级从 L2 提升到 SM 级，缓解"BSK 热数据打爆 L2 带宽/延迟"的瓶颈（baseline 中 stall_long_scoreboard >50%）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 分块与合并访问的协同（图 7，Para-B：N=512、k=4、ℓ=2）：分块几何取 8 个连续复数 FP64 元素（16B/个 = 128B），对齐 GPU 128 字节内存事务粒度，保证每块 TBSK/TGLWE 读天然合并（coalesced），且不改变 FFT 输出的数据布局（避免显式重排的额外开销）：
```
# 单个线程块处理：BSK 的一个 tile（128B）× 一批 GLWE 的对应 tile
for tile_idx in 0..t-1:                    # t = 总 tile 数
    tbsk = load_tile(BSK, tile_idx)         # 128B 合并读，被整批 GLWE 复用
    for glwe in batch:                      # 同参数的一批 PBS 密文
        tglwe = load_tile(FFT(glwe), tile_idx)
        acc[tile_idx] += tbsk ⊙ tglwe       # Hadamard 积（逐元素乘累加）
# 多个线程块并行覆盖不同 tile → 一次 kernel 启动处理大量 GLWE MAC
```
- Annotations：tile 大小是带宽与复用度的权衡——32/64/128B 都满足合并访问（现代 NVIDIA 内存事务最优 128B），更小 tile 提高 BSK 复用因子但增加循环/指令开销，更大 tile 提升 ILP；经优化后 BSK 访问带宽不再是主要瓶颈（相对傅里叶系数访问），故 MNEMOS 取 128B（8 元素）最大化 ILP。k 越大（安全级别依赖 kN，Concrete 常用大 k）BSK 足迹占比越高、复用收益越大（消融 +MAC 1.10×~1.77×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CUDA kernel 内按 tile 索引加载 BSK 与 GLWE 傅里叶系数并做分块 Hadamard 乘累加；配合 Tensor Core FFT 的连续输出布局（无需重排）。使用场景：TFHE 批量 PBS（加密 CNN 逐层、AES 等 192 独立输入）；batch 越大收益越稳（图 15：baseline 在 batch>1024 因工作集超 40MB L2 而性能骤降，MNEMOS 全程稳定）。效果：GMEM→L2 流量降 15.7%、L2→SM 降 69.4%，stall_long_scoreboard 从 >50% 降至 ~20%。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
