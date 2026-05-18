## VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

- baseline方法是什么？
  Baseline是CPU/GPU上朴素VAR（Visual Autoregressive）推理：VAR每次迭代仅decode一个visual token，一张256×256图像需256至4096次串行Transformer invocation。每次invocation内完整执行attention（O(N²)复杂度）和FFN，不利用图像token的空间冗余和不同Transformer层间的计算冗余。MaskGIT类parallel decoding使用启发式mask schedule但质量和硬件友好性不足；speculative decoding类方法引入draft model额外开销且仅2-3 token/step并行度。ViTCoD、AdapTiV等ViT ASIC accelerator仅对attention或FFN做单侧冗余利用，仍为串行解码。

  全栈执行例子（以Vanilla VAR 256×256 + V100 GPU为例）：
  - 算法层：VAR自回归next-token prediction——每step选1个visual token→完整transformer forward（QKV attention O(N²) + FFN，所有token参与计算）→写回→下一step——重复256次。注意力图随着层数加深趋于相似但未利用此特性。单张256×256图像生成常需10-60秒。
  - 系统框架层：论文未明确说明（V100单卡PyTorch推理，无多请求调度）。
  - 编译框架层：论文未明确说明（PyTorch默认编译路径）。
  - kernel调度层：GPU上标准attention和FFN kernel（cuBLAS/cuDNN），无定制dataflow。TopK排序使用通用Bitonic Sort+Merge Sort。
  - 硬件架构层：Intel Xeon Platinum 8168 CPU、NVIDIA V100 GPU；ASIC baseline ViTCoD/AdapTiV（28nm）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文从信息论角度刻画VAR模型的双重冗余（Dual Redundancy）：(1) Image Redundancy——图像token的空间相关性远高于语言token（entropy/redundancy分析），使跨迭代并行解码成为可能；(2) Model Redundancy——attention可解释为低通滤波，随着层数加深，深层attention map趋于相似形成"模型惯性"。基于此提出三项算法优化+两项硬件dataflow。

  **缺陷1（Inter-Iteration）：每次仅解码1个visual token，数百至数千次串行Transformer调用导致高延迟**
  → PD (Draft-Free Parallel Decoding)：无需draft model，在每轮中对所有masked token位置预测并按置信度TopK一次性解码多个token。原理：图像token比语言token有更高的空间相关性，熵值更低，因此可直接用模型自身的置信度判断而非依赖draft model。PD-aware training选择sampling temperature、masking ratio r(t)、guidance scale。256×256下仅需8-32步（减少>80%），一轮最多解码64 token。

  **缺陷2（Intra-Iteration - 浅层）：浅层attention全局计算存在大量token间相似性冗余**
  → TA (Token Aggregation)：利用attention作为低通滤波的insight——浅层仍处于"Learning Region"，但局部窗口内token高度相关。将token分local window→Small Attention（OP dataflow）将窗口内token聚合为representative token→Big Attention（Row dataflow）对浓缩后的代表token做全局建模。减少41% attention MAC，质量下降<0.5%。

  **缺陷3（Intra-Iteration - 深层）：深层attention map趋于相似、token重要性分化，但baseline对所有token执行完整计算**
  → DB (Dynamic Bypass)：深层"Inert Region"中注意力趋于均匀化，大部分token信息增量低。轻量MLP对每token打分→Radix Sort Core选TopK重要token→仅这些token进入完整attention+FFN→被bypass token通过token restoration（Token_i × JudgeWeight_i + Token_i）将原有信息补回下一层输入。额外减少58% MAC。schedule function控制逐层skip rate（α=0.3, β=-0.4, max=0.55）。

  **缺陷4（硬件）：TA引入Small/Big Attention异构执行模式，PD和DB需大K TopK（N=4096时K=1936），通用排序方案在大K上延迟高（TopK仅3.5%操作但占20.9%延迟）**
  → Unified Attention Core：同一PE array通过Snooper+Fat Tree动态切换Row（Big Attention）和OP（Small Attention）dataflow，避免为两类attention放独立core造成低利用率。
  → Radix Sort Core：将大K TopK转为固定4阶段流水线（CountBin→PrefixSum→SelectBin→Filter），加Locality-aware Scheduling根据mask history优先调度高置信区域，消除全局排序的反复读写重排开销。

  论文方法全栈执行例子（以VAR-Turbo-Balance 256×256 8步为例）：
  - 算法层：VQGAN tokenization→全masked V0。每轮：Transformer预测所有masked token→PD Gumbel sampling+置信度TopK→释放K(t)个token→mask更新。浅层0-15层TA（Small Attention local window→representative token→Big Attention全局），深层DB（MLP打分→TopK→attention+FFN→bypass token restoration）。8步完成（vs baseline 256步），FID 2.67 vs baseline 2.65。
  - 系统框架层：论文未明确说明。
  - 编译框架层：论文未明确说明。
  - kernel调度/硬件架构层：28nm VAR-Turbo accelerator (7.09mm², 1.98W) + HBM2 32GB/s。Unified Attention Core Row/OP dataflow+Radix Sort Core 4-stage TopK+MLP/Non-Linear/SIMD Cores层间pipeline。vs V100 GPU 210.3× speedup, 423.5× energy efficiency improvement。
