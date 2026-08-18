## csegfold（cycle-accurate SpGEMM 加速器模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
csegfold 是 SegFold 论文自研的 cycle-accurate C++ 模拟器，忠实建模 SegFold 微架构与 Segment 动态数据流：SELECTA 调度（A bitmask 扫描、active window）、SEGMENTBC（merge network 比较/转发/插入/累加）、IPM 二叉搜索与 LUT 更新、spatial/temporal folding、memory controller（coalescing、cache/HBM2 时序）、tiling。构建时经 CMake FetchContent 自动拉取 Ramulator 2.0 作为 offchip HBM2 内存后端，所有硬件组件逐周期推进。评估输出 simulated cycles 与相对 baseline（Spada/Flexagon）的 speedup；面积/功耗由 RTL 综合给出（非模拟器）。全部硬件组件在 cycle-by-cycle 基础上模拟以确保时序精度。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
模拟流程：输入 = 稀疏矩阵（SuiteSparse MatrixMarket ~20 个 + 合成矩阵，密度 0.05–1.0、尺寸 256–1024）+ 硬件配置（Table II：16×16 PE、1 GHz、window 32、cache 1.5 MiB/16-way/128B、HBM2-8Gb 2Gbps、4 路 vector multicast）→ 内存控制器逐周期跑 SELECTA 选 (m,k) → B 行经 vector multicast + row shifter 注入 → merge network 中每个 B 元素按 b/c 比较定位/插入/累加 → C 驻留 PE 或折叠/spad 溢出 → Ramulator2 计时 HBM2 请求 → 输出每矩阵 cycle 计数与访存统计 → 归一化为 speedup 并绘图（Fig.8–12）。确定性：模拟器完全确定，CSV 输出应与 expected_results/ 逐位一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源（GitHub https://github.com/PolyArch/SegFold-AE + Zenodo DOI 10.5281/zenodo.19453259，MIT）。使用：Ubuntu 22.04+、GCC 10+（C++20）、CMake ≥3.15、Python ≥3.8（numpy/scipy/matplotlib/pandas/pyyaml）、可选 Docker；硬件需求 ≥4 核 CPU（推荐 16+）、≥64 GB RAM（推荐 256 GB，breakdown/mapping 消融单进程可达 50 GB）、2–5 GB 磁盘。流程：`cmake -B build && cmake --build build -j` 编译 csegfold（Ramulator2 自动拉取）→ `python3 scripts/download_matrices.py` 下载 SuiteSparse 矩阵 → `./scripts/setup.sh` 构建+冒烟测试 → 运行实验产出 per-experiment CSV（fig8/9/10/11/12 对应结果）→ plot_overall.py 等生成 PDF/PNG。实验规模 209 个 simulation runs，16 核约 2 小时；Docker 容器经 `docker compose run artifact ./scripts/run_all.sh` 一键复现。RTL 侧：SystemVerilog 源码 + 综合报告随仓库分发。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
