## Swizzled Shared Memory Layout (NVIDIA Hopper)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Swizzled Shared Memory Layout 是通过对 shared memory 地址进行 XOR 变换来重排数据元素到不同 memory bank 的布局技术，消除 bank conflict（多线程同时访问同一 bank 导致串行化）。NVIDIA Hopper shared memory 由 32 个 bank 组成（4 字节宽），bank conflict 显著增加访问延迟。ThunderKittens 提供三种编译时自动选择的 swizzle：32 字节（4-way conflict, width≤16 的 tile）、64 字节（2-way conflict, width≤32）、128 字节（0 conflict, width≤64 且 bf16）。与 row-major（8-way conflict when loading to tensor core layout）和 padded 布局（无 conflict 但地址非对齐，不兼容 TMA/HGMMA）相比，swizzle 在消除 conflict 的同时保持地址对齐，兼容 H100 的 TMA bulk copy 和 WGMMA 指令。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
128 字节 swizzle 实现：
```
bf16* swizzled_layout_128B(bf16 *data, int r, int c) {
    uint64_t addr = (uint64_t)&data[r * columns + c];
    return (bf16*)(addr ^ (((addr % (128*8)) >> 7) << 4));
}
```
原理：取地址对 128×8=1024 字节的余数，右移 7 位，左移 4 位得 XOR 值（16 字节粒度位翻转），与原始地址 XOR——使同一列不同行的连续元素映射到不同 bank。

TK 编译时自动选择：
```
if (tile_width <= 16 && type==bf16)  → 32B swizzle  // 4-way conflict
elif (tile_width <= 32)              → 64B swizzle  // 2-way conflict
else                                  → 128B swizzle // 0 conflict, WGMMA/TMA compatible
```
NCU profiling 证据：FlashAttention-3 (CUTLASS) 存在 9.6-way bank conflict → shared memory stall 0.92 cycles；TK attention kernel 的 shared memory stall 仅 0.14 cycles —— 85% reduction，归功于自动 swizzle 选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 中用户无需手动选择 swizzle——定义 `st_bf<H, W>` shared tile 时编译器自动根据 W 和数据类型选择最优布局。自动 swizzle 是 TK 对比 CUTLASS 的核心优势之一：CUTLASS 需要程序员手动管理 shared memory layout（经常导致保留的 bank conflict），TK 将 layout 作为框架内部优化自动化。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels
