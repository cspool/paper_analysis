## MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  MixFusion的kernel/运行时核心是Patch Edge Stitcher和Compressed Sparse Patch (CSP)格式。关键设计：(1) Patch Edge Stitcher：将跨patch边界stitching操作fuse进GroupNorm kernel，消除额外memory movement。具体实现：每个GPU thread block (TB) normalizes一个patch的同时检查其boundary pixels是否被邻接patch需要→需要的boundary pixels暂存于shared memory→所有normalization完成后TB定位目标patch并将boundary写回global memory。此设计overlap了edge stitching与normalization，无需额外synchronization；(2) Fused GroupNorm+Stitcher kernel：TB内通过shared memory暂存boundary data实现column方向的irregular memory access局部化（column stitch按行方向内存布局时访问不连续→通过shared memory中转化解），row boundaries直接按内存layout对齐高效读写；(3) CSP格式：受CSR格式启发，通过resolution reorder→offset-based compression将patch mapping压缩为RequestOffset[]和RequestStart[]/RequestEnd[]三数组，O(1)定位任意patch所属请求及其邻接关系；(4) Batched Cache Operations：将patch级cache的query/delete/update/insert操作coalesce为batch——输入patch indices与cache entries比对→分出Common Set（需verification）/New Set（insert）/Expired Set（delete）→三集合并行处理。实验对比：(a) Patch Edge Stitcher vs. naive stitching latency overhead（Figure 5, stitcher overhead minimal vs. naive offsetting parallelism gains）；(b) PSNR/SSIM quality vs. Distrifusion across patch sizes（Table 4, Patch Size=512用4 Patches PSNR 28.82/SSIM 0.88）；(c) patched vs. whole-image caching latency savings（Figure 20, patch-level consistently outpeforms full image）；(d) cache overhead vs. batch size（Figure 17, cache overhead scales modestly）。

- 后端平台是什么，配置是什么。
  NVIDIA H100-80GB GPU (CUDA 12.3, PyTorch 2.2.2)，xformers用于加速attention算子。

- 评估性能的软件/脚本是什么。修改了什么。
  基于PyTorch + custom C++/CUDA实现（12.5K行总代码）。新增kernel/运行时：(1) Patch Edge Stitcher CUDA kernel：fused GroupNorm + boundary stitching，TB-level shared memory通信实现irregular memory access局部化；(2) CSP格式运行时：python-side resolution reorder + offset计算 + C++ tensor indexing；(3) CuML Random Forest用于GPU端cache prediction（避免CPU-GPU数据搬移开销）；(4) xformers集成用于加速Self-Attention中的batched attention。实验中对patch splitting overhead和cache management overhead分别ablation测量（Figure 17），patch size对throughput影响的sensitivity analysis（Figure 18, patch size 64/128/256）。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/desenSunUBW/mixfusion。kernel使用流程：
  1. Patch Edge Stitcher kernel：当patched convolution执行时，每个TB处理一个patch的GroupNorm→TB检查自身boundary pixels是否需要被邻接patch使用（依赖信息在patch splitting时记录）→需要的boundary存入TB的shared memory→完成所有normalization→TB根据metadata定位需要其boundary的目标patch→将boundary从shared memory写回global memory对应位置→目标patch的TB可直接读取准确边界值
  2. CSP查找：给定patch ID→二分查找RequestOffset[]确定所属请求→RequestStart[patch_id]和RequestEnd[patch_id]获取该patch在所属请求内的起止位置→ResolutionOffset[request_index]提供Self-Attention reconstruction时的分辨率偏移
  3. Batched Cache：block收到patch indices和intermediate results→cache system以map结构（patch unique ID为key）比对→Common Set中patch比较cached vs. new result决定update→New Set合并为batch insert→Expired Set（cache中存在但input indices无）的patch已退出→batch delete

MixFusion kernel/运行时的作用：通过fused Patch Edge Stitcher消除跨patch计算中的冗余memory movement（边stitching边normalization，利用shared memory localize irregular column access），通过CSP格式实现O(1) patch定位和高效批量cache操作，确保patch-level parallelism的计算收益不被management overhead侵蚀。

