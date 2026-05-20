## Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Streaming Multilevel Concentration的运行时计算调度，将稀疏压缩嵌入systolic-array accelerator的GEMM tile执行流中。核心调度设计：(1) SEC Streaming top-k overlapping：SEC的a-way streaming bubble sorter执行M·a·k cycles，与image attention GEMM (M·(M+T)·h·n/(a·b) cycles)完全重叠。在典型配置下(h·n≈3584, b=32, k<M+T)，sorting操作远在Q(i)K^T完成前结束，SEC不在critical path上。(2) SIC on-chip tile-local compression：SIC不等整层或全序列token就绪，在每个GEMM m×n tile产生后立即on-chip压缩（m=1024, n=32, a=n=32），每个tile最多8×m cycles做similarity matching（7 pairwise comparisons + 1 L2-norm per vector）vs GEMM需要K/b×m=112×m cycles (K=3584, b=32)，matching远不在critical path。仅当K<256时matcher接近critical path，此时可scale多matcher并行。(3) Convolution-style layouter conflict-free scheduling：按Bank=f%2×4+r%2×2+c%2映射使2×2×2 block的8 vectors分散到8个不同SRAM bank，支持无复制无bank conflict的全并行读取。Offset=⌊r/2⌋×⌈W/2⌉+⌊c/2⌋计算地址。(4) Scatter-Gather循环：Similarity Scatter用2a-wide accumulator (64)做concurrent accumulation，根据similarity map将compact vectors的partial sums复制/distribute回原始token indices在output-stationary buffer中累加；完成所有⌈K/k⌉ outer loop iterations后Similarity Gather做一次性tile-level再次压缩。(5) SEC-to-SIC handoff：SEC的offset encoding随GEMM output stream传输给SIC，使SIC的convolution-style layouter可恢复prune后token的(Frame,Height,Width)坐标。实验比较：performance speedup和energy，对比vanilla SA、AdapTiV、CMC、GPU (A100)、GPU+FrameFusion。Focus实现4.47× speedup vs SA、7.90× vs GPU、2.37× vs GPU+FrameFusion。DRAM traffic分析：CMC虽有46% sparsity但仍有79% dense DRAM traffic，Focus达81% sparsity且仅需21% bandwidth。

- 后端平台是什么，配置是什么。
  Focus accelerator：32×32 PE array (FP16 multiply/FP32 accumulate, weight stationary dataflow)。On-chip buffer 734KB (128KB input + 78KB weight + 512KB output + 16KB layouter buffer for 256-vector window)。Off-chip memory: DDR4 4Gb×16, 2133R, 4 channels, 64GB/s。Target clock 1.32ns (≈757 MHz)，500MHz place-and-route目标下34% timing margin。TSMC N28HPC+工艺。GPU对照：NVIDIA A100 80GB (FP16), Jetson Orin Nano GPU。

- 评估性能的软件/脚本是什么。修改了什么。
  基于SCALEsim-v2构建cycle-accurate simulation framework。输入：PyTorch实现生成的layer-wise sparse traces（记录每GEMM tile的active/inactive token indices、similarity map、concentrated vector count）。修改SCALEsim-v2：添加Focus-specific runtime scheduling建模（SEC top-k sorter与attention GEMM的重叠调度、SIC similarity gather/scatter per tile的pipeline staging、convolution-style layouter bank conflict检测、scatter accumulator的2a-wide concurrent operation建模）。DRAM energy使用DRAMsim3建模。RTL (SystemVerilog) 用Synopsys Design Compiler综合。所有baseline accelerator的核心逻辑同样用SystemVerilog实现以公平评估面积和能耗。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/dubcyfor3/Focus（MIT License, algorithm/simulator/rtl/evaluation_scripts）。Zenodo DOI: https://doi.org/10.5281/zenodo.17851346。评估使用流程：
  1. 算法trace生成：在A100 GPU上运行PyTorch Focus实现，对LLaVA-Video-7B/LLaVA-OneVision-7B/MiniCPM-V-2.6在VideoMME/MVB/MLVU上推理，record per-layer per-tile的sparse traces（SEC token pruning decisions + SIC similarity map + concentrated vector indices）
  2. Simulator运行：`python simulator/run.py --config focus_arch.yaml --traces <trace_dir>` → 模拟32×32 PE array上GEMM tiling + SEC streaming top-k + SIC scatter/gather per tile → 输出cycles breakdown (GEMM/SEC/SIC/idle)、DRAM read/write bytes、on-chip buffer utilization
  3. RTL评估：`cd rtl/ && make synth` → Synopsys DC综合 → area/power report；Memory Compiler生成SRAM macros
  4. 对比baseline：同样配置下simulate vanilla SA/AdapTiV/CMC → compare speedup/energy/DRAM traffic
  5. 环境：Ubuntu 22.04, Python 3.11, PyTorch 2.6.0, CUDA, Transformers 4.48.2/4.49.0, FlashAttention 2.7.4.post1, A100 80GB

