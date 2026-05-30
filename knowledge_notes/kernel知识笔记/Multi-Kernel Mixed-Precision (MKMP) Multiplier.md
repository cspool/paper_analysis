## Multi-Kernel Mixed-Precision (MKMP) Multiplier

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Kernel Mixed-Precision (MKMP) Multiplier是Squat (EdgeQAT)论文提出的SIMD-based混合精度矩阵乘法器，用于在移动设备上高效执行sub-8-bit混合精度MAC操作。核心设计：将现有的INT8 multiplier与自定义INT4 multiplier整合到同一个GeMM kernel中，由Token Control Logic Module (TCLM)根据每个token的位宽动态路由到对应multiplier。INT4 multiplier基于INT8 multiplier构建，通过将相邻行4-bit权重拼接存入16-bit寄存器，利用ARM `mla`指令（32-bit目标寄存器INT32）在单条指令内完成乘加。理论上4-bit GEMM的计算操作数减半（vs 传统零扩展到8-bit）。MKMP multiplier解决了两个关键挑战：(1)标准SIMD INT8 kernel不支持混合精度；(2)sub-8-bit数据需零扩展至byte边界，浪费计算能力。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MKMP multiplier在移动CPU上的执行流程（以LLaMA-58M W4A8(1:1)推理一个token为例）：
```
// TCLM: Token分组
float scores[N];  // attn[:, 0]
int k = N * rho;
int threshold = heapsort_topk(scores, k);

// 分组8-bit和4-bit tokens
vector<float> x_8_group, x_4_group;
for (int i = 0; i < N; i++)
    if (scores[i] >= threshold) x_8_group.push_back(x[i]);
    else x_4_group.push_back(x[i]);

// === INT8 Multiplier (处理8-bit tokens) ===
// 标准ARM NEON vmlaq_s8() SIMD指令
// x8_int8: INT8 quantized activations
// w_packed: INT4 packed weights (offline quantized)
int8x16_t x8 = vld1q_s8(x8_int8_ptr);
int8x16_t w8 = vld1q_s8(w8_ptr);  // 4-bit权重先解包为8-bit
int32x4_t acc8 = vmlaq_s32(acc8, x8, w8);  // 乘加

// === INT4 Multiplier (处理4-bit tokens, INT4 Concatenation) ===
// Step 1: 加载相邻两行4-bit权重拼接为16-bit
// w_row_i: 4-bit, w_row_i+1: 4-bit -> concat: [w_row_i | w_row_i+1] in 16-bit
uint16x8_t w_concat = load_concat_4bit_weights(w_ptr);

// Step 2: 4-bit激活加载
uint8x16_t x4 = vld1q_u8(x4_int4_ptr);

// Step 3: 16-bit宽乘加 (ARM mla: 16-bit x 16-bit -> 32-bit acc)
// 内部拆分保持数学精度
int32x4_t acc4 = int4_concat_mma(x4, w_concat, acc4);

// Step 4: Bit-shift + row-wise summation
acc4 = vshlq_s32(acc4, shift_amounts);
acc4 = row_wise_sum(acc4);

// === 合并结果（反量化） ===
output_8 = dequantize(acc8, alpha_x8, alpha_w);
output_4 = dequantize(acc4, alpha_x4, alpha_w);
output = concat_and_reorder(output_8, output_4);  // 恢复原始token顺序
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MKMP multiplier的INT4部分基于gemmlowp/QNNPACK的INT8 GEMM kernel修改。INT4 concatenation利用ARM `mla`指令的32-bit目标寄存器特性（更宽的INT32累加器可容纳更多部分和而不溢出）。Compiler-level优化：分配计算线程时考虑不同操作的内存读取模式，重叠内存读取时间。移动CPU上实测加速：LLaMA-58M OnePlus 11 W4A4=2.24× vs FP16，GPT2-97M Raspberry Pi 5 W4A4=2.37× vs FP16。混合精度W4A8(1:1)在Raspberry Pi上额外加速超40%（vs pure W8A8）。代码开源：https://github.com/shawnricecake/squant。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

---
