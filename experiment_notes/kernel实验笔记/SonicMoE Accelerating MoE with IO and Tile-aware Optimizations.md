## SonicMoE Accelerating MoE with IO and Tile-aware Optimizations

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SonicMoE 是基于 CuTe-DSL 编写的 GPU kernel 库，面向 NVIDIA Hopper (H100) 和 Blackwell (B300) GPU，为细粒度、高稀疏 MoE 训练提供 8 个高性能 kernel：(1) **Forward：Up-proj A kernel**——Gather + varlen-M Grouped GEMM + SwiGLU，将 token gather 操作与 GMEM-to-SMEM load 融合（cp.async），SwiGLU 融合到 GEMM epilogue；(2) **Forward：Down-proj Y kernel**——varlen-M Grouped GEMM + 异步 TMA store，使用 Ping-Pong scheduling（Hopper）或 TMEM 两阶段流水线（Blackwell）将 MMA 与 heavy epilogue IO 重叠；(3) **Forward：Expert aggregation O kernel**——每个 token gather-and-sum 所有激活 expert 的输出，基于 TMA gather 实现高带宽；(4) **Backward：dH kernel**——Gather dO + varlen-M Grouped GEMM + dSwiGLU + dS计算 + A'输出，将 Gather fusion、epilogue fusion（dH/dS/A'同一 kernel）和异步 TMA load of H 全部融合；(5) **Backward：dW2 kernel**——Gather dO + varlen-K Grouped GEMM；(6) **Backward：dX~ kernel**——varlen-M Grouped GEMM + 异步 TMA store；(7) **Backward：dW1 kernel**——Gather X + varlen-K Grouped GEMM；(8) **Backward：dX kernel**——每个 token gather-and-sum 所有 expert 的 dX~。关键设计：(a) **Gather Fusion**：在 forward 和 backward 的所有需要 gather 的地方均将 gather 与 GMEM-to-SMEM load 融合，消除 X_e 和 dO_e 的显存物化；(b) **Epilogue Fusion**：SwiGLU/dSwiGLU/dS 均融合在 GEMM epilogue 中，dS 使用 dS=⟨dA',A⟩ 路径（而非 ⟨dO,Y⟩），避免缓存 Y 和额外 HBM 访问；(c) **Ping-Pong Scheduling**（Hopper）：2 consumer warpgroups 交替执行 MMA 和 epilogue/IO，实现 MMA 与 IO 重叠；(d) **TMEM 两阶段流水线**（Blackwell）：利用 UMMA 的单线程异步特性和 TMEM 的 2-stage 结构，MMA warp 与 epilogue warps 并发操作不同 TMEM stage；(e) **Token Rounding Routing**：将 per-expert token 数舍入到 GEMM tile size (M_tile=128) 的倍数，消除 Grouped GEMM padding 带来的浪费 FLOPs。
  - 实验比较：(a) **Kernel 级 TFLOPS**：SonicMoE vs ScatterMoE/MoMoE/MegaBlocks/Megatron/DeepGEMM++/DeepGEMM-pt，在 H100 和 B300 上测 forward+backward 的 TFLOPS 和 memory bandwidth；(b) **Activation Memory**：各方法 per-layer peak activation memory（1.4B~120B MoE 配置），SonicMoE 减少 45%（7B, n=256 vs ScatterMoE）；(c) **端到端训练吞吐**：7B MoE (n=256) FSDP-2 训练，SonicMoE 64 H100s = 213B tokens/day vs ScatterMoE 96 H100s = 225B tokens/day；(d) **Grouped GEMM 基础性能**：contiguously-packed inputs 和 gathered inputs 的 varlen-M/K Grouped GEMM TFLOPS vs DeepGEMM/cuBLAS；(e) **Token Rounding 训练吞吐**：TR vs TC top-K 在稀疏 MoE (E=128/256) 下的 TFLOPS 对比；(f) **Top-K sorting kernel**：SonicMoE bitonic sort top-K vs PyTorch/Triton/Tilelang/RTop-K 的带宽对比；(g) **Expert aggregation kernel**：gather-and-sum vs scatter-and-sum 策略的带宽对比；(h) **Ablation**：Gather fusion 有无、Ping-Pong scheduling 有无、TMA store vs st.global scatter 对 TFLOPS 的影响。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA H100 (Hopper, SM90) 80GB 和 NVIDIA B300 (Blackwell, SM100)。H100 使用 CUDA toolkit v12.9；B300 使用 CUDA toolkit v13.0。
  - 多节点训练：64× H100（8 nodes × 8 GPUs）使用 FSDP-2 + ZeRO-3 单节点内 shard，节点间复制。
  - 软件框架：CuTe-DSL（CUTLASS 的 C++ template DSL）编写 kernel，PyTorch 接口。基于 lm-engine 代码库进行端到端训练。

- 评估性能的软件/脚本是什么。修改了什么。
  - SonicMoE 开源：https://github.com/Dao-AILab/sonic-moe（permissive license）
  - 基础框架：CUTLASS CuTe-DSL + PyTorch。对比 baseline：ScatterMoE (Triton)、MoMoE (Triton)、MegaBlocks (block-sparse)、Megatron-LM GroupedMLP (CUTLASS Grouped GEMM)、DeepGEMM (SM90/SM100 BF16 Grouped GEMM)
  - 修改/新增内容：(1) 实现 8 个独立 MoE kernel（forward A/Y/O + backward dH/dW2/dX~/dW1/dX），使用 CuTe-DSL 的 warp-specialized kernel 设计；(2) 实现 Gather fusion with cp.async 在 varlen-M 和 varlen-K Grouped GEMM 中；(3) 实现 Ping-Pong scheduling（Hopper）和 TMEM-based overlapping（Blackwell）；(4) 实现 top-K bitonic sorting kernel（支持 E≤4096, K≤16）；(5) 实现 Token Rounding routing 算法；(6) 实现高效 expert aggregation kernel（TMA gather-and-sum）
  - 快速使用：`pip install sonic-moe`，提供 PyTorch nn.Module 接口直接替换 MoE 层

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/Dao-AILab/sonic-moe，论文 ICLR'26（https://openreview.net/pdf?id=KzTJ1raEgB）
  - 评估原理：kernel 级 benchmark 使用 CUDA event 计时测量每个 kernel 的 wall-clock time，计算 TFLOPS = (6+12)dn(Σf_e) / time（model FLOPs 而非 hardware FLOPs）。Memory bandwidth = total IO bytes / time。端到端训练吞吐使用 lm-engine 代码库测量 tokens/day。
  - Kernel 执行全过程（以 dH kernel 为例，H100，7B MoE, n=256）：
    ```
    输入: dO ∈ R^{T×d}（上游梯度）, W_2 ∈ R^{E×n×d}（down-proj weights）, S（router scores）, π（routing mask）, H（cached pre-activation）
    
    [Prologue - TMA + cp.async]
    Step 1: Producer warpgroup 启动 TMA load W_2 到 SMEM pipeline stage
    Step 2: Consumer warpgroup 0 启动 WGMMA，accumulate 到 RF
    Step 3: Producer warpgroup 启动 cp.async + gather 加载 dO（按 π 索引 gather）
    
    [Mainloop - Ping-Pong]
    Step 4: Consumer warpgroup 0 继续 WGMMA over K dim
    Step 5: Consumer warpgroup 1 启动 WGMMA，利用另一组 accumulator registers
    Step 6: Producer warpgroup 交替为两个 consumer 提供下一 tile 数据
    // Consumer 0 和 1 交替执行 MMA 和 epilogue
    
    [Epilogue - Consumer warpgroup 0]
    Step 7: MMA 完成，dA'_e = dO_e W_{2,e}^T 结果在 RF 中
    Step 8: 异步 TMA load H_e 从 HBM 到 SMEM（dedicated pipeline）
    Step 9: 从 SMEM 读取 s_e = Gather(S, π_{:,e}) → dA_e = Broadcast(s_e) * dA'_e
    Step 10: dSwiGLU(dA_e, H_e) → dH_e, A_e（forward activation recompute）
    Step 11: A'_e = Broadcast(s_e) * A_e（write to HBM via TMA store, 用于 dW2）
    Step 12: dS_{e,t} = ⟨dA'_{e,t}, A_{e,t}⟩（reduce over n dim）
    Step 13: TMA store dH_e, dS, A'_e to HBM
    
    // Consumer warpgroup 1 同时执行下一个 tile 的 MMA
    
    输出: dH ∈ R^{TK×2n}（up-proj activation grad）, dS ∈ R^{T×E}（router score grad）, A' ∈ R^{TK×n}（输入给 dW2 kernel）
    ```
  - 8 个 kernel 的工作流：Forward: X → [A kernel: Gather+GEMM+SwiGLU] → H,A → [Y kernel: GEMM+TMA store] → Y → [O kernel: Gather-and-sum] → O。Backward: dO → [dH kernel: Gather+GEMM+dSwiGLU+dS+A'] → dH,dS,A' → [dW2 kernel: Gather+GEMM] → dW2；dH → [dX~ kernel: GEMM+TMA store] → dX~ → [dW1 kernel: Gather+GEMM] → dW1；dX~ → [dX kernel: Gather-and-sum] → dX。
