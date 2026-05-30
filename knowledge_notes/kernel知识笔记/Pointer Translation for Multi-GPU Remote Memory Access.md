## Pointer Translation for Multi-GPU Remote Memory Access

术语是什么？
Pointer Translation是Iris实现跨GPU transparent memory access的核心算法——利用symmetric heap的同构性（所有GPU的heap在相同偏移处存储对称数据），将本地指针转换为目标GPU上的等价虚拟地址。翻译步骤：(1) 从heap_bases数组加载源/目标基址 → (2) `offset = ptr - from_base` → (3) `remote_ptr = to_base + offset` → (4) cast回指针类型。heap_bases数组（8 GPU × 8 bytes = 64 bytes）在kernel执行期间常驻L1 cache，翻译开销实测为零（被通信延迟完全主导）。

从kernel调度角度拆解术语：
```
@triton.jit
def __translate(ptr, from_rank, to_rank, heap_bases):
    from_base = tl.load(heap_bases + from_rank)     # GPU_src heap base
    to_base = tl.load(heap_bases + to_rank)          # GPU_dst heap base
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base                     # 同构heap → 相同offset
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_ptr = to_base_byte + offset
    return tl.cast(translated_ptr, ptr.dtype)

@triton.jit
def load(pointer, to_rank, from_rank, heap_bases, mask=None):
    translated_ptr = __translate(pointer, to_rank, from_rank, heap_bases)
    return tl.load(translated_ptr, mask=mask)  # 直接跨GPU load
```
翻译仅需~5条指令（两次tl.load + 减法 + cast），所有Iris device-side API调用前自动执行。Triton编译器可优化翻译指令与后续remote操作之间的pipeline。

术语一般如何实现？如何使用？
开发者无需手动调用__translate——Iris的load/store/get/put/copy/atomic_* API在内部自动完成翻译。仅需传入heap_bases作为所有device-side API的必需参数。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
