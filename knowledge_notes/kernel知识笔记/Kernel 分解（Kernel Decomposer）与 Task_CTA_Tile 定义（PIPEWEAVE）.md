## Kernel 分解（Kernel Decomposer）与 Task/CTA/Tile 定义（PIPEWEAVE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Kernel Decomposer 是 PIPEWEAVE（ISCA'26 GPU 性能预测框架）的第一个模块，把一次 kernel 启动的完整工作量拆成一组基本调度单元 task：`{τ1,...,τt} = F(X, S)`，其中 X 是 kernel 输入参数、S 是硬件规格。task 的精确定义随执行范式变化——传统 GPU 执行模型（如 FlashAttention-2）中 task=CTA（Cooperative Thread Array，即 thread block），一次 launch 生成 CTA grid，硬件调度器把每个 CTA 分给一个 SM；persistent kernel 模型（如 Ping-Pong GEMM、FlashAttention-3）中 CTA 长期驻留 SM 作为 worker，真正的基本调度单元是从全局 work queue 取的更小计算包 tile。每个 task 用维度参数向量 d_i（如 GEMM 的 tile 几何 {tile_M, tile_N, tile_K}）刻画规模；causal attention 下即使是"名义相同"的 task，实际工作量也因掩码而不同。与 Neusight/Habitat 从 profiling 数据反推简化 tiling 不同，PIPEWEAVE 对开源 kernel（FlashInfer、SGLang、vLLM）直接读源码提取并行化与 thread block 映射逻辑得到确定性的 F。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 BF16 GEMM (M×N×K) 为例，若 kernel 用 tile (128,128,64) 且是常规（非 persistent）实现：
```
# 伪代码：Kernel Decomposer 输出 task 集合
grid_m = ceil(M / 128); grid_n = ceil(N / 128)
tasks = []
for i in range(grid_m):
    for j in range(grid_n):
        tasks.append(Task(d = {tile_M: 128, tile_N: 128, tile_K: 64,
                                origin: (i*128, j*128)}))   # 一个 CTA = 一个 task
# 后续 Scheduling Simulator 把 tasks 按 RR/软件 tile 调度分到各 SM
```
对 cuBLAS 这类闭源 kernel，F 无法从源码提取，改用经验推断：用 PyTorch Profiler 在大量 (M,N,K) 组合上 profiling，分析 kernel 名、CTA 数与输入尺寸的相关性，反推出代理（surrogate）映射函数近似其隐式 task 切分；unseen GPU 上闭源 kernel 借用架构最相近 GPU 的切分逻辑。各 kernel 的 Decomposer 实现仅 10–50 行代码。论文验证：分解出的 CTA 数与数据集 ground-truth 完全一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上开源 artifact 位于 https://github.com/zksainx/pipeweave，analytical_model/ 目录为每类算子（GEMM/FA2/FA3/RMSNorm/SiLU×Mul/MoE）提供计算器。使用流程：给定 kernel 类型与输入维度，Decomposer 产出 task 集合→Scheduling Simulator 按调度范式（硬件 RR 或软件 tile scheduler）把 task 映射到 SM 得到 task 分布→Feature Analyzer 按每 pipeline 汇总 demand/理论周期→MLP 预测执行效率。该分解同时保证 kernel 泛化性：任何 kernel 都被转成统一 task 分布，与来源无关，因此新 kernel 只需写一个 Decomposer。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
