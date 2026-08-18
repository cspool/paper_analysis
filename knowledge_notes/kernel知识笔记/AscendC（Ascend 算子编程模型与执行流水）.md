## AscendC（Ascend 算子编程模型与执行流水）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
华为 CANN（Compute Architecture for Neural Networks）框架内的 C++ 算子编程模型，用于在 Ascend NPU 上开发高性能自定义算子（2023 年 CANN 7.0 起替代旧 TBE 框架）。核心抽象：tensor（封装将由 AIC/AIV 操作的数据）与 queue（同步机制——操作完成 EnQue，依赖操作 DeQue），配合 Pipe 管理资源；标准三段式执行范式 CopyIn → Compute → CopyOut（数据经 MTE 搬进片上缓冲 → 计算单元执行 → 结果搬出），多流水（vector/cube/mte）可重叠。Ascend 把每个 AI core 当作单一重线程（无 CUDA 式轻量线程间同步），靠任务队列驱动的流水数据流重叠数据搬运与计算。ENEC 论文用 AscendC（C++17）实现压缩/解压算子，并指出其限制：32 字节对齐约束、无 gather/scatter、无条件分支、整数算术指令少。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ENEC 压缩 kernel 的 AscendC 流水骨架（每 AIV 线程处理 16384 元素块，循环分发）：
```
// CopyIn: MTE 把 HBM 权重块搬进 UB
Copy(ub_in, gm_weights, block);
// Compute（AIV 向量单元，全逐元素指令）:
  E = ExtractExponent(ub_in);            // 拆分 BF16 指数
  y = (E - b) * -1 & mask;               // 分支无关整数变换
  packed = HierarchicalHalvingPack(y, m, L);  // 位宽量化 + lane folding 打包
  EnQue(packed_q, packed);               // 结果入队
// CopyOut: 满 32KB buffer 后输出低 16 位 + bit mask 到压缩流，右移继续
DeQue(next_q);                            // 取下一块数据
```
Annotations：EnQue/DeQue 队列抽象让搬运-计算-搬运三段重叠；无分支意味着所有"如果超过 m 位"的判断都提前用 bit mask 表达；每 core 单线程但 48 个 AIV 并行处理不同块。编译：CANN toolkit（AscendC 编译器）把 C++ 编成 AI core 指令，产物为 .so 算子。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AscendC 语言 + CANN 工具链（Ascend-cann-toolkit 8.2.RC1.alpha002 + kernels-910b 同版本），Ubuntu aarch64 + Python 3.9 + torch/torch_npu；build_csrc.sh 编译 csrc/ 目录 NPU kernel 为 .so；msprof 做 kernel 级 profiling。使用：任何 Ascend 自定义算子开发（ENEC 压缩/解压算子、模型算子）；开发者用 tensor/queue/pipe 表达数据流，靠 tiling 把数据切进 UB（ENEC 选 16384 元素块，32K 会超 UB 192KB）。局限（ENEC 论文强调）：无分支/无 gather/无变长内存操作/无轻量同步——这正是传统无损压缩算法在 Ascend 上"根本性不兼容"的根源。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
