## Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

- 属于硬件架构的实现是什么？实验比较什么？
  提出Focus Unit硬件模块，集成到systolic-array accelerator的memory interface附近，在GEMM/attention/FC执行流中做on-chip streaming concentration。Focus Unit包含两个子模块：(1) Semantic Concentrator (SEC)：importance analyzer用并行max units处理attention SoftMax输出的text-to-image cross-modal attention scores（a个max units并发），a-way streaming bubble sorter做top-k selection（M·a·k cycles vs attention GEMM的M·(M+T)·h·n/(a·b) cycles，完全与图像attention GEMM重叠），offset encoder用sliding window为保留token记录相对位置offset（lightweight registers）。(2) Similarity Concentrator (SIC)：Similarity Gather在GEMM tile输出后做vector-level cosine similarity matching（dot-product unit + L2-norm precompute buffer），convolution-style layouter按Bank=f%2×4+r%2×2+c%2映射使2×2×2 block的8 vectors落在不同SRAM bank实现无conflict并行访问，similarity map (1×m)记录每个原始vector到compact buffer中代表vector的映射；Similarity Scatter用2a-wide accumulator (64 when a=32)做concurrent accumulation恢复原始token位置。Focus Unit总on-chip buffer 734KB（128KB input + 78KB weight + 512KB output + 16KB layouter buffer for 256-vector window）。实验比较：performance (speedup)、energy、area、power、DRAM traffic，对比vanilla systolic array (SA)、AdapTiV、CMC、GPU (A100)、GPU+FrameFusion。Focus面积3.21 mm² (TSMC 28nm)，功耗736 mW，相比SA仅增加2.7% area和0.9% power；SEC占1.9%总面积，SIC占0.8%。平均4.47× speedup vs SA，2.60× vs AdapTiV，2.35× vs CMC，7.90× vs GPU，2.37× vs GPU+FrameFusion。能效4.67× vs SA，17.09× vs GPU，2.98× vs AdapTiV，3.29× vs CMC。DRAM traffic减少4.9× vs dense SA，2.2× vs AdapTiV，3.7× vs CMC。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  基于SCALEsim-v2 ([51])构建cycle-accurate simulation framework。SCALEsim-v2为开源 systolic-array simulator (https://github.com/Hieu0155/SCALE-sim-v2)。DRAM能耗使用DRAMsim3 ([38], 开源 https://github.com/umd-memsys/DRAMsim3)建模。On-chip SRAM由TSMC N28HPC+ Memory Compiler生成。RTL (SystemVerilog) 用Synopsys Design Compiler在28nm SS corner (0.81V, 125°C)综合。

- 模拟器模拟什么的性能，修改了什么。
  Simulator接受layer-wise sparse traces（从PyTorch实现中特定模型和数据集生成），建模cycles和memory access。修改SCALEsim-v2：添加Focus Unit模块的cycle-accurate建模，包括SEC importance analyzer/sorter/offset encoder的延迟、SIC similarity gather (cosine matching + similarity map generation)/scatter (partial sum redistribution + accumulation)的延迟、convolution-style layouter的bank conflict建模、on-chip buffer read/write timing。DRAM traffic由simulator追踪output write-back和input fetch的access pattern自动计算。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：https://github.com/dubcyfor3/Focus（含algorithm、simulator、rtl、evaluation_scripts，MIT License）。Zenodo DOI: https://doi.org/10.5281/zenodo.17851346。模拟器使用流程：
  1. Algorithm trace generation：`python algorithm/generate_traces.py --model Llava-Video-7B --dataset VideoMME` → PyTorch实现运行VLM推理，record每层layer-wise sparse traces（每GEMM tile的active/inactive token indices、similarity map、concentrated vector count）
  2. Simulator配置：设置Focus硬件参数（PE array 32×32, m=1024, n=32, block 2×2×2, threshold=0.9, semantic ratios per layer）和DRAM配置（DDR4 4Gb×16, 2133R, 4 channels, 64GB/s）
  3. Cycle-accurate simulation：`python simulator/run.py --config focus_config.yaml --traces traces/` → simulator逐层模拟GEMM tiling + SEC (top-k sorting overlapped with attention GEMM) + SIC (similarity gather/scatter per tile) → 输出per-layer cycles、DRAM read/write traffic、on-chip buffer utilization
  4. Area/Power：RTL (`rtl/`) 用Synopsys DC综合 (`make synth`) → 28nm SS corner 1.32ns target → 报告area 3.21 mm²和power 736 mW
  5. 环境要求：Ubuntu 22.04, Python 3.11, PyTorch 2.6.0, CUDA, Transformers 4.48.2/4.49.0, FlashAttention 2.7.4.post1, A100 80GB HBM, ~128GB storage
