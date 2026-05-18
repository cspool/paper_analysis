## TCA-TBE (Tensor-Core-Aware Triple Bitmap Encoding)

术语是什么？通过联网搜索让回答具体和精准。
TCA-TBE是ZipServ提出的面向GPU Tensor Core的固定长度无损压缩编码格式。其核心思路是：利用LLM中BF16权重指数字段的高偏态分布（>95%权重仅使用top-7连续exponent），将每个8×8 weight tile编码为三个独立64-bit bitmap（每bit表示codeword的一个bit-plane）加两个紧凑value buffer（PackedSignMantissa和FullValue fallback），替代Huffman/ANS等变长entropy codec。3-bit codeword使平均每元素仅11.3 bits（理论下界10.6 bits），压缩率约1.41×。TCA-TBE是对同一团队此前SpInfer中TCA-BME（面向sparsity的bitmap encoding）的演进，将bitmap方法从稀疏模式扩展到无损压缩场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TCA-TBE算法pipeline分两阶段：
1) 离线压缩（Algorithm 1 in paper）:
   - 对每层weight matrix扫描exponent直方图 → 选top-7连续exponent值（覆盖>95%权重），记录base_exp = min(top_exponents) - 1
   - 对每个8×8 FragTile编码：遍历64个元素，若exponent∈top-7则计算3-bit codeword c = exponent - base_exp（c∈[1,7]），将c的三个bit分别写入三个bitmap，将sign+mantissa (8-bit)写入PackedSignMantissa buffer；若exponent∉top-7，将完整BF16写入FullValue fallback buffer
   - 输出：三个全局bitmap array + PackedSignMantissa array + FullValue array + Offset array
2) 在线解压（per-thread）:
   - 对每个元素，bitwise OR三个bitmap得到64-bit spatial indicator mask M
   - 若M[position]=1（高频）：读取PackedSignMantissa中的sign+mantissa，从bitmap恢复3-bit codeword c，exponent = base_exp + c（implicit lookup，无表查找），组装BF16
   - 若M[position]=0（fallback）：直接从FullValue读取完整BF16
解压全程仅需bitwise OR + POPC（population count）+ integer ADD，无分支、无shared memory table lookup。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TCA-TBE实现为CUDA/C++（约2.5K行），作为ZipServ的offline compressor模块。使用方式：
1. 加载BF16模型权重 → 逐层调用compressor分析exponent分布 → 生成TCA-TBE格式压缩文件
2. 压缩LLaMA3.1-8B约需2.5分钟（16核CPU），为一次性offline操作
3. 运行时由ZipGEMM kernel直接读取TCA-TBE格式进行fused decompression+GEMM
TCA-TBE的3层tiling设计（8×8 FragTile → 16×16 TensorCoreTile → 64×64 BlockTile）直接对齐NVIDIA Tensor Core的mma.m16n8k16 operand register layout，消除runtime坐标变换。开源地址：https://github.com/HPMLL/ZipServ_ASPLOS26.git

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

