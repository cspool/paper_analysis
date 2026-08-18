## 统一内存架构与多应用带宽争用（Unified Memory Architecture / Bandwidth Contention）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 移动 SoC（如 Snapdragon 8 Elite）采用统一内存架构：CPU、GPU、NPU 共享同一个受带宽约束的系统内存（LPDDR5），各处理单元的有效可用带宽会随并发运行负载动态波动。SMOOTH（ISCA'26）把它作为"系统引发的运行期带宽波动"（System-Induced Bandwidth Variability）动机：在多程序执行下，NPU 可用的 idle 带宽随 CPU/GPU 活动的有无与类型大幅变化（论文用 Geekbench 6 在 Galaxy S25+ 上测量 2 个 CPU + 4 个 GPU 负载组合下的 NPU idle 带宽），这种波动使静态编译器难以选择匹配运行期带宽的 tile size——这是编译期静态优化无法覆盖的运行期系统条件。
从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：用户同时跑聊天应用（NPU 上的 LLM）+ 视频（GPU 解码）+ 前台交互（CPU）→ 三者争用统一内存带宽 → NPU 看到的内存带宽忽高忽低 → 静态编译器在编译期固定的 tile size/预取调度与实际带宽不匹配 → GEMV 层带宽饱和时 stall、softmax 等 compute 阶段带宽又空闲。SMOOTH 的硬件 DMC 在运行期动态测量可用带宽（N_preload = ⌊U×BW/Block_size⌋ 中 BW 由硬件测量），按当前争用状况决定预取多少块，从而适应波动带宽；评估中在 64GB/s 最大带宽 + Geekbench co-run 干扰下，SMOOTH-ER 较 Compiler-Ideal 平均 ITL 增益 42.7%、较 SMOOTH-Base 5.0%。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：移动 SoC 硬件的统一内存是平台属性（SMOOTH 不修改），SMOOTH 在硬件内做带宽测量与自适应预取；模拟中通过配置 DRAM 带宽（16/32/64/128 GB/s）与 Geekbench co-run 工作负载干扰来复现争用。相关概念：系统层面的带宽争用感知调度（如 Fusa 争用感知分发策略）与硬件侧的动态带宽测量（SMOOTH 的 DMC）属于不同层次但互补。论文未明确说明 serving/多请求调度层（batch=1、单请求 on-device 场景）。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
