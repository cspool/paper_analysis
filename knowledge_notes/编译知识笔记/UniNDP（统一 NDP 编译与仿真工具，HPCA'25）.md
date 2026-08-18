## UniNDP（统一 NDP 编译与仿真工具，HPCA'25）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UniNDP 是清华 NICS 组在 HPCA 2025（pp. 624–640，DOI 10.1109/HPCA61900.2025.00054）提出的统一 NDP 编译与仿真工具：① 统一的树形 NDP 硬件抽象 + 对应指令集，支持基于不同 DRAM 技术的多种 NDP 架构（Channel/Rank/Device/Bank 层级划分）；② cycle-accurate、指令驱动的 NDP 仿真器，跟踪内存单元与 PU 工作状态；③ NDP 编译器，优化数据划分、映射与负载调度；④ 硬件状态引导的搜索空间裁剪 + 基于 DRAM 时序参数的快速性能预测器。相对既有映射/编译方法取得 1.05–3.43× 加速。开源：https://github.com/thu-nics/UniNDP（HPCA'25 artifact 另有 UniNDP-hpca25-ae/UniNDP）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
FlexQ-NDP 把 UniNDP 作为基座做三处修改：① 在原指令格式外包一层携带量化元数据的高层 IR（如 QGroup 尺寸、scale 位置），支撑去量化隐藏等优化；② 仿真器侧把这类指令转换为 DRAM bank/PU/缓冲命令并逐 cycle 仿真 scale 读取、dequant 与分组部分和；③ 编译器侧建模 scale 缓冲与 partial-sum 缓冲，向算子 kernel 插入相应指令。运转流程：FlexQ-NDP 编译产物（带量化元数据 IR）→ UniNDP 展开为 ACT/PRE/RD/WR 等 DRAM 命令（按 DRAMSim3 时序推进）与 MAC/MUL PU 命令 → 逐 cycle 统计行切换、缓冲命中、PU 忙闲 → 输出延迟与活动 trace（能量按 DRAMSim3 功率模型折算）。artifact 实验即用 bash scripts/final/3_single_op_with_predictor/ 系列脚本驱动（DSE→逐 config 仿真→理论下界）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Python 实现的开源工具（FlexQ-NDP artifact：https://github.com/ISCA26-FlexQ-NDP-ae/flexq_ndp，MIT License，Zenodo DOI 10.5281/zenodo.19452117；pip install -r requirements.txt + export FLEXQ_NDP_DIR）。使用：作为 NDP 架构研究的标准实验床——评估算子划分、数据布局、指令调度策略对 NDP 性能的影响；扩展方式是加 IR 元数据与新 pass（FlexQ-NDP 即此路径）。限制：仿真规模受 Python cycle 模拟速度约束（大 MM 算子单 config 仿真需数小时，artifact 提供 Zenodo 日志备份）。

NASZIP 补充视角（ISCA'26，UniNDP 作为周期精确 NDP 模拟器）：NASZIP 修改 UniNDP 支持 rank-level parallelism（README："UniNDP is modified to support rank level parallelism"），在其上配置 DDR5 多通道/多 rank/双 sub-channel 结构（DDR5-4800、2/6 通道、2 DIMM/通道、2 rank/DIMM、2 VPE+LNC/rank、256KB LNC-D、8KB LNC-T、1.2GHz），逐 cycle 模拟 ANNS 搜索的 QPS、延迟分解（邻居检索/距离计算/部分结果处理）与 recall，配合 Synopsys DC 28nm 综合 + Cadence Innovus P&R 评估面积/功耗、3D-ICE 评估热。仿真驱动脚本：fee_dim_freq.sh / overall_basic.sh / overall_hp.sh / qps_vs_recall_*.sh / prefetch_hit_rate.sh / cache_hit_rate.sh → result/sync_csv_results.py → Plot/plot_fig_*.py。开源于 NasZip 仓库 UniNDP/ 目录（https://github.com/Intelligent-Computing-Research-Group/NasZip，Apache-2.0）。

涉及论文标题：
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
