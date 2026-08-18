## DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking（近似层次匹配：无 chiplet/存储介质级结构优化，本层取其 3D 堆叠 + hybrid bonding 的物理结构实现）

- 属于芯片设计的实现是什么？实验比较什么？
  - 实现为 3D 堆叠图像传感器的物理结构：顶层像素阵列（PSC 模拟前端：光电二极管 + 开关电容放大器的对数式光强变化检测）与底层逻辑层（SSPL + PAC 阵列 + 外围）经 hybrid bonding 键合，仅两条键合信号垂直互连——Vdiff 数据通路与 SCtrl 控制通路；事件在底层完成就地采样、patch 计数、AER 打包，输出 buffer 经 MIPI CSI-2 送出。工艺结构：像素模拟电路 40 nm CMOS（对数放大器/源跟随器 2.5 V、其余像素电路 1.1 V），数字电路 22 nm（0.8 V）综合后经 DeepScaleTool 折算到 40 nm，模拟器件与 SRAM cell 直接用 40 nm 库。
  - 实验比较：底层像素面积/组成对比标准 DVS 像素 [117]（36.97 vs 36.76 μm²，握手逻辑 21%→18%、事件总线驱动 buffer 15%→10%、采样逻辑 6%→15%）；整芯片面积 3.414 mm²（346×260 阵列，顶层像素 5 μm、底层像素 6.1 μm，SRAM 外围 0.067 mm²，握手逻辑与输出 buffer 开销 <0.1% 忽略）；系统级功耗-延迟对比 3D-EPV/3D-PV/2D-V/BlissCam/TinyTracker（详见硬件架构层条目）。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 无专用芯片结构/热应力模拟器：物理/版图层用 Cadence Virtuoso（40 nm 版图 + SPICE 仿真），数字层用 Synopsys Design Compiler + ARM Memory Compiler（22 nm 综合、DeepScaleTool 折算 40 nm）；论文未修改既有模拟器、无公开链接。
- 模拟器模拟什么的性能，修改了什么。
  - Virtuoso 模拟像素模拟电路与版图（面积、触发延迟、静态/动态功耗），DC/Memory Compiler 给出 PAC 数字逻辑与 SRAM 外围的面积/功耗；3D 堆叠与 hybrid bonding 的工艺可行性按堆叠图像传感器文献 [56,109,135] 论证（键合信号仅两条，代价极低），未做热/应力/键合良率级仿真。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？
  - 开源：论文未说明芯片设计 RTL/版图开源。使用例子：顶层 PSC 沿用 [48,117] 设计（5×5 μm²），检测光强变化生成 DC 电压 Vdiff，经 hybrid bonding 传到底层 SSPL 的两个比较器（与 VH/VL 阈值比较生成 ON/OFF 事件写 2-bit SDP SRAM）；每像素仅增加 1 个 6T + 2 个 8T SRAM 单元，16×16 像素共享一个 PAC 握手单元与加法树（免 per-pixel AER 握手电路），显著压缩底层面积；整个阵列按 patch 寻址（22×17 个 patch），AER 包（addrX/addrY 各 5 bit + 512 bit 事件 + 32 bit timestamp）经 ping-pong 行 buffer、输出 FIFO、MIPI CSI-2 输出。顶层模拟前端主导像素面积，底层 ESS/PAC 逻辑总开销仅 0.6%，验证"3D 堆叠 + 像素级稀疏化"的物理可行性；未来工作可在底层继续集成专用 NPU 替代片外 STM32N6x7。
