## ML 模型权重解压（decompress+execute，压缩权重执行流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对稀疏化 + 量化的 ML 模型，推理时在线把压缩存储的权重 tile 解压成稠密矩阵再交给矩阵单元计算（"decompress+execute"）。存储层用稀疏（结构化/非结构化）+ 量化（低位宽）压缩权重以省内存与带宽；计算层解压出的 tile 直接喂给核内矩阵单元（如 Intel AMX/TMUL）。ATX 论文把它作为第四个评测 kernel：DECA-like NCA 从内存读压缩 tile、解压后写回核寄存器，核立即用 AMX 对解压 tile 做 GeMM——这是核与加速器**双计算**、细粒度交错的代表用例（此前三个 kernel 都是加速器主算、核只做控制）。任务输入仅 512B–2KB，任务产出被下一环节（AMX 指令）直接消费。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
流水线（论文图 19 场景，按压缩因子 CF 扫描）：
```
for tile in model_weights:                  # 权重按 tile 分块
    task = {VAccId, compressed_tile_addr, tile_shape, CF}
    ATX V1T2(task) → NCA(DECA-like) 解压   # 输出进 1-2 个 1KB tile 寄存器
    AMX_TMUL(decompressed_tile, activations) # 核立即消费解压结果
```
执行节奏：NCA 解压任务与核 AMX 计算交错进行——一个 tile 解压的同时上一个 tile 在做 TMUL；解压任务小（512B–2KB 输入）意味着高频的核↔加速器往返，调用开销成为关键：论文测得 ATX NCA 较 core-only 4.0×、ICA 1.8×、L2 OCA 3.9×、LLC OCA 18×（18× 正是小任务下 OCA 串行调用 + fence 开销的放大）。软件基线是 libxsmm 的 decompress+execute kernel（AVX512 + AMX）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
生产形态：权重稀疏化（如 2:4 结构化稀疏）与量化（INT8/INT4）后，推理库（libxsmm、PyTorch 后端）在计算前做反量化/稠密化；专用硬件如 DECA（MICRO'25）做近核解压器，配合 TEPL ISA 扩展支持乱序调用隐藏通信延迟，并配 3D roofline（Roof-Surface）性能模型。ATX 的使用方式：解压任务经 ATX 指令与 UTE 流引擎调度，解压结果直接进 tile 寄存器供 AMX 消费，省去"解压写内存 → 再读回"的往返。适用条件：压缩权重模型推理、带宽敏感（HBM）平台、任务粒度小且与核计算紧耦合；若解压本身可完全批量离线完成，则不需要在线流式交错。GPU 变体（Approaching Shannon Bound 论文）：把该模式升级为"压缩权重执行原语"——rANS 熵编码 tile 常驻全局内存，解码 warp 按 GEMM tiling 序解压直接写 shared memory（不落全局内存）、GEMM warp 经 tensor core 立即消费，双缓冲流水重叠；基线对照为 NeuZip/DFloat11 的层粒度 decompress-store-compute（整层解压写回全局内存再 GEMM，无重叠、有层同步屏障）。

涉及论文标题：
- ATX: Accelerator Task Extensions
- Approaching Shannon Bound with Lossless LLM Weight Compression
