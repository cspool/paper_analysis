## Accelerating Sparse Transformer Inference on GPU (STOF)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Unified MHA Module，包含row-wise和block-wise两种customized GPU kernel实现sparse MHA计算。关键设计：(1) Two-level sparse storage format：外层OuterTile (OT) 为64个8×8 InnerTile (IT)，用BSR (Block Compressed Sparse Row) 表示OT级稀疏——full OTs通过full_row_ptr/full_col_idx数组定位，part OTs通过part_row_ptr/part_col_idx定位+bitmap_mask (64×uint64) 表示IT内64个元素的精确mask pattern；(2) Block-wise kernel (Algorithm 1)：Q按OT_Size_M切分子块Q_i保持在register中，K/V按OT_Size_N切分子块K_Tj/V_j，根据load_row_ptr和load_col_idx仅加载需要计算的valid OTs，跳过无效块；async data copying (__async_memcpy)使V加载与GEMM重叠；对part OTs使用bitmap_mask做细粒度mask；(3) Row-wise kernel：Q按row sliced，warp内使用shuffle操作通信，消除warp间synchronization，适合小seq_len+高稀疏率场景；(4) Advanced optimizations：8×8 IT对齐Tensor Core mma数据粒度、OT行主序存储适配Softmax迭代计算、IT列主序存储消除bank conflict、Q_i register resident避免重复SMEM读写；(5) Kernel selection analytical model：公式1基于valid OT ratio和seq_len计算threshold，低于threshold选row-wise否则block-wise。实验比较MHA computation performance（RTX 4090/A100，causal/sliding window/Longformer/Bigbird，seq_len 128-4096，batch size 1/8/16），对比PyTorch Native、FA2、FlexAttention、ByteTransformer、MCFuser、SPLAT。STOF相对FlexAttention在RTX 4090上平均1.8×、A100上平均1.6×加速。sliding window上（93.8% sparsity）加速最显著，(batch=16, seq_len=4096)时STOF达到FA2的4.8×、FlexAttention的4.9×。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 4090 (Ada Lovelace, 24GB, 128 SM)、NVIDIA A100 (Ampere, 80GB, 108 SM)、NVIDIA H20 (Hopper, preliminary test)。CUDA v12.6, PyTorch 2.7.0。FP16精度。

- 评估性能的软件/脚本是什么。修改了什么。
  NVIDIA Nsight用于profiling。kernel通过CUDA/C++实现（约2,500 LOC），基于FA2的CuTe结构扩展，引入two-level storage format和对应优化。Kernel通过torch/cpp_extension接口封装为PyTorch native function，首次调用时ninja JIT编译为.so动态链接。实验100次warm-up + 100次timed iterations取平均。修改：SPLAT论文未开源，基于论文内容复现。所有方法统一FP16精度评估。MHA实验遵循BERT-Base配置。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：Zenodo artifact DOI: 10.5281/zenodo.17705801。STOF MHA kernel使用流程：
  1. Mask预处理：将任意mask pattern转换为two-level storage format——划分OT/IT网格→生成full_row_ptr/full_col_idx（full OTs的CSR索引）→生成part_row_ptr/part_col_idx（part OTs的CSR索引）→对每个part OT生成64×uint64 bitmap_mask
  2. Kernel Selection：analytical model（公式1）输入valid OT数量（通过load_row_ptr计算）和seq_len，输出threshold，小于0选row-wise kernel否则block-wise kernel
  3. Block-wise kernel执行（以BERT-Base, Bigbird mask, seq_len=4096为例）：
     - Q切分子块Q_i→保持在register→外层循环遍历Q_i
     - 内层循环：load_row_ptr确定当前row的valid OT数量→load_col_idx获取列索引→cp.async加载K_Tj→__async_memcpy加载V_j（与GEMM重叠）→Compute_GEMM(Q_i, K_Tj)得P_ij→检查part OTs→若为part OT则Apply_Mask(P_ij, bitmap_mask)做bitwise mask→Softmax→Compute_GEMM(S_ij, V_j)累加O_i→write back to HBM
  4. Row-wise kernel（小seq_len+高稀疏）：Q row sliced→warp内shuffle同步→集中式mask处理利用row locality
  5. 例如sliding window mask (93.8% sparsity)在A100上(batch=16, seq_len=4096)，block-wise kernel跳过绝大多数OT计算→STOF加速FA2 4.8×

