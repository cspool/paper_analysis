## Compressed Sparse Patch (CSP) Format

术语是什么？通过联网搜索让回答具体和精准。

Compressed Sparse Patch (CSP) Format是MixFusion提出的一种受Compressed Sparse Row (CSR)启发的patch管理数据结构。CSR是稀疏矩阵的经典压缩存储格式（三数组：values/column_indices/row_offsets），CSP将其思想迁移到图像patch管理场景：将不同分辨率请求产生的heterogeneous patches通过resolution reorder + offset-based compression高效存储和定位。核心差异在于CSR的block size固定且位于矩阵左上角，而CSP支持diverse block sizes（各resolution产生不同数量patches）且patches按请求连续分布。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

CSP格式的内存布局与查询流程：

```
内存数组（以4请求, resolutions=[1024,512,768,1024], patch_size=256为例）:
  RequestStart[] = [0,0,0,0, 4,4,...,4, 13,13,...,13, 17,17,...,17]
                   // 每个patch对应其所属请求的起始patch ID
  RequestEnd[]   = [3,3,3,3, 12,12,...,12, 16,16,...,16, 32,32,...,32]
                   // 每个patch对应其所属请求的终止patch ID
  ResolutionOffset[] = [0, 8, 17, 33]
                   // 累计patch数：4+9+4+16=33(实际), 用于Self-Attn分组
  RequestOffset[] = [0, 4, 13, 17]  // 每个请求的首patch在全局数组中的偏移

查询操作：
  patch_id → 二分查找RequestOffset确定所属请求request_idx
  request_idx → RequestOffset[idx]到RequestOffset[idx+1]遍历该请求所有patches
  request_idx → ResolutionOffset[idx]提供Self-Attention reconstruction偏移

构造流程：
1. 请求按arrival时间排序 (Figure 9a)
2. 按resolution重排序→patches视为sparse array (Figure 9c)
3. 仅记录每个patch的request和resolution metadata
4. 计算累积offset：RequestOffset（per-request首patch偏移）
5. 计算分辨率偏移：ResolutionOffset（per-request resolution分组偏移）
```

CSP的O(1)定位能力来自offset预计算：任何patch通过其global index二分查找RequestOffset即可确定所属请求，再通过RequestStart/End确定在请求内的patch范围。这比naive per-patch metadata存储（需记录resolution/shape/offset等）大幅节省内存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CSP在MixFusion中作为Python + C++ runtime实现。patch splitting时根据batch内各resolution的GCD确定patch_size→计算每个请求的N_h×N_w patches→按resolution排序后算出所有offset arrays→denoising过程中所有patch操作通过CSP提供的O(1) lookup定位上下文（如Convolution边界stitching所需的邻接patch、Self-Attention全图reconstruction所需的分组信息）。CSP的设计使patch管理overhead modest（Figure 17中splitting overhead仅占latency的极小比例），且支持dynamic batch composition（新请求到达时只需重新计算offset arrays，无需重组patch数据）。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

