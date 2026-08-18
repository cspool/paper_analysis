## Bringing Near Data Processing into the Low-Bit Floating-Point Era

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现为面向分组低比特 FP 的 NDP 硬件扩展 + 多精度 PU RTL：(1) 在 Hynix GDDR6-AiM [33] 架构基础上扩展为支持 group-wise scale 与 dequant 的低比特 FP NDP：32 芯片×2 通道×16 bank，每 bank 挂一个多精度 PU + 5Kb（20×32B）SRAM，主机经 PCIe 通信，256b 操作数宽度对齐 GDDR6 读写位宽；(2) 多精度点积 PU：FP32 的 24-bit 尾数乘法器分解为 4 个 12-bit、12-bit 再分解为 4 个 6-bit，低精度（FP4/FP8）获得更高并行度；加法树按层降精度——第一级 FP8、第二级 BF16、更高层 FP32；支持 fp4(E2M1)/fp8(E4M3)/fp16(E5M10)/fp32(E8M23)，尾数位 2/4/11/24，每 DRAM 列存放 64/32/16/8 个值；PU 另支持无累加 MUL 操作用于 weight-only 的 dequant；(3) 设计权衡：单周期 FP32（0.0185 mm²、8.72 mW）vs 2/4 周期 FP32（0.0167/0.0152 mm²、7.21/6.04 mW），选用单周期 FP32 设计点，0.4 GHz 频率保证一个 MAC 在一个 bank 访问间隔（t_CCDL×t_CK）内完成。RTL 用 Synopsys DC Compiler 于 14nm（对齐 GDDR6-AiM 的 1y-nm DRAM 工艺）综合评估面积/功耗。
  - 实验比较：灵敏度实验——SRAM 缓冲容量（baseline 需 >20×32B 才接近最优，FlexQ-NDP 8×32B 即接近最优，缓冲需求降约 3×；缓冲增大后加速比仍稳定在 1.38×）、FP32 吞吐（1/2、1/4 吞吐下平均 1.45×/1.28× 加速）、混合精度 PU 假设（FP16×FP4 结构，FP16 吞吐 4× 至 51.2 GFLOPS，W4A16S8 下 baseline 降 1.66×、FlexQ-NDP 平均 3.05×）。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 修改 UniNDP [63]（https://github.com/thu-nics/UniNDP，HPCA'25 统一 NDP 编译+仿真工具，cycle-accurate）；DRAM 时序/功率参数取自 DRAMSim3 [38]；PU 面积/功耗用 Synopsys DC Compiler（14nm）综合。
- 模拟器模拟什么的性能，修改了什么。
  - 模拟 NDP 上低比特 FP GEMM 的 cycle-accurate 延迟与能量。修改：① 在 UniNDP 指令格式外包一层携带量化元数据的高层 IR；② 仿真器支持把这些指令转换为 DRAM bank/PU/缓冲命令并逐 cycle 仿真 scale 读取、dequant 与分组部分和；③ 编译器侧建模 scale 缓冲与 partial-sum 缓冲以插入相应指令。FlexQ-NDP 的开源仿真框架即此修改版 UniNDP。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 开源：https://github.com/ISCA26-FlexQ-NDP-ae/flexq_ndp（MIT License，Python+Shell），结果文件 Zenodo DOI https://doi.org/10.5281/zenodo.19452117；安装后 export FLEXQ_NDP_DIR 并从项目根目录运行 scripts/final/ 下脚本。模拟原理：输入 = 硬件描述（Tab. V：bank/PU/缓冲容量、频率、时序）+ 编译出的指令流 → 指令展开为 DRAM 命令（ACT/PRE/RD/WR，按 DRAMSim3 的 tRCD/tRP/tCCDL 时序推进状态机）与 PU 命令（MAC/MUL）→ 逐 cycle 统计行切换、缓冲命中、PU 忙闲与 dequant 空闲 → 输出 cycle 延迟（与 PU 无空闲理论下界归一对比，Fig.8）→ 能量由 trace 按 DRAMSim3 功率模型计算（Tab. VI）。PU RTL 侧：Verilog 经 DC 14nm 综合得面积/功耗表（Tab. II），用于验证低比特 FP 支持与单周期设计点开销。
