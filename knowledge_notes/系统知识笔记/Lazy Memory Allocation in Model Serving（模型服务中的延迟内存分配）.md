## Lazy Memory Allocation in Model Serving（模型服务中的延迟内存分配）

术语是什么？
Lazy Memory Allocation 是 MoEsaic 为 vLLM 设计的内存管理优化，用于解决 expert deduplication 场景下的内存峰值问题。vLLM 原在模型初始化时为所有 expert 预分配 GPU 显存，即使用于容纳多个 model instance 的内存总量可能超过 GPU 容量——即使去重后会释放大量内存，预分配阶段的内存峰值仍会导致 OOM。MoEsaic 用 tiny pseudo experts（伪专家，仅占极小显存）初始化所有 expert 结构，在实际加载参数时才逐步扩容（resize）并填充，确保最大显存占用 = 去重后的模型大小。

从系统架构角度拆解术语：
Lazy Allocation 在 MoEsaic 中的执行流程：

1. **伪初始化**：模型初始化时，所有 expert 用 tiny pseudo tensors（如 shape=(1,1) 的零张量）占位，预分配总显存极小。
2. **逐步扩容**：从模型文件逐 tensor 加载参数。每个 expert 完全填充后 resize 至实际 shape（如 gate_proj: (14336, 4096), up_proj: (14336, 4096), down_proj: (4096, 14336) for Mixtral-4x7B）。
3. **去重检查**：expert 完全填充后计算 hash → 查 dictionary → 若命中则释放新分配的显存并引用已有 tensor。
4. **显存峰值**：峰值出现在"当前正在加载（尚未去重）的一个 expert + 所有已去重 expert"的时刻，而非"所有 model instance 的所有 expert 展开"的时刻。
5. **内存安全**：只要 GPU 能容下去重后的模型 + 1 个完整 expert，加载就不会 OOM。

术语一般如何实现？如何使用？
- 类似 Linux 的 copy-on-write 思想——先装"假的"，需要时再分配。不同的是 MoEsaic 不涉及写时复制，而是直接用 hash 检测相同性并共享。
- 在 vLLM 中实现为修改 `load_weights()` 方法：用 placeholder tensors 替换预分配的 full-size tensors，加载时逐步替换。
- 主要优势：使得"GPU 仅需容纳去重后模型"成为可能（而非去重前 N× 模型大小），是多 instance 部署的前提条件。
- 论文表 1（Table 1）：Mixtral-4x7B, 4 model instances, 2 shared experts——去重前需 224 GB（8×40GB GPU），去重后仅 140 GB（4×40GB GPU），节省 84 GB。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts
