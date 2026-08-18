## ANT 加速器模拟器（ANT simulator，ant_simulator）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ANT（Adaptive Numerical Type, MICRO'22，SJTU 开源 https://github.com/SJTU-ReArch-Group/ANT_MICRO22，Apache-2.0）提出 flint 自适应数值类型 + TypeFusion PE，其配套开源的 ant_simulator 是验证过的加速器性能/能量仿真器。EVA 基于 ANT 模拟器构建自研仿真器：所有 baseline 加速器（SA、ANT、FIGNA、FIGLUT）集成进同一模拟器与 EVA 公平对比，模拟 LLM（dense + MoE）推理的 latency/energy/power/area/throughput。硬件侧另用 Verilog HDL 实现（EVA 与全部 baseline）+ Cadence Genus（TSMC 28nm @500MHz）综合测面积/功耗，Cacti 7.0 建模 SRAM，DRAMsim3 模拟 DRAM 功耗。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
EVA 仿真器输入→输出：YAML study 配置（simulator/configs/studies/，指定 models/methods/sequence lengths/batch sizes/study 参数，CLI 可 --models/--methods/--scenarios 覆盖）+ 预处理 trace（模型结构、量化配置、访问模式）→ 按 500MHz/64GB/s 配置建模 DRAM 流式读 input tile（v×d=32×8）与 WI（v×N）→ 32×8 FP16 PE 阵列 VQ-GEMM（256 cycles）产出 OC → OC 片内直送 4 个 EU 冲突无关查找+加法树归约 → 与下一 GEMM tile 重叠 → output tile 写回 DRAM → 输出 CSV（latency/energy/throughput）到 simulator/output/ → notebook（notebooks/hardware_results.ipynb）渲染论文 Fig.8-14/Table III,VIII-X。9 个硬件仿真 study（复现 Fig.8-14 + Table III,VIII-X），约 2 小时，无需 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：EVA 仓库 https://github.com/dbw6/Eva.git（MIT）simulator/ 目录（main.py、configs/studies/、traces/、output/），scripts/run_simulator_parallel.sh 并行跑；Zenodo 归档 https://doi.org/10.5281/zenodo.19433707；依赖 numpy/pandas/pyyaml/matplotlib。使用方式：无需 GPU 的 x86-64 CPU（≥16GB RAM、约 10GB 磁盘）；改 YAML 或 CLI 参数做 DSE（EU 数、VQ 配置 n/C/d、batch 大小）；硬件仿真复现 Fig.8-14 与 Table III,VIII-X，算法精度评估（eval_ppl.py/lmeval.py）另需 A100-80GB。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
