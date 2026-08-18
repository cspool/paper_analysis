## 逻辑错误率分层累加模型（Layer-wise Logical Error Rate Model，LER 分层模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LER 分层模型是评估晶格手术执行质量的度量模型（O3LS 采用 SPARO [28] 的方法）：把整个执行按时间片（time slice / layer）划分，每层的逻辑错误率由三类独立错误源复合，再逐层线性累加得到总 LER：
  $$p_{\text{total}} \approx \sum_{t=1}^{T} p_{\text{layer}}^{(t)} \approx \sum_{t=1}^{T} \big(1 - (1 - P_{\text{PPM}}^{(t)})(1 - P_{\text{PR}}^{(t)})(1 - P_{\text{idle}}^{(t)})\big)$$
  假设稀有失败与独立错误事件。P_PPM=Pauli 乘积测量错误率（主要由路由空间与码距决定）、P_PR=patch rotation 错误率（分解为变形/角移动/移动三片分别仿真）、P_idle=idle 记忆错误率。该模型使"时间步 vs ancilla 路径长度"的 trade-off 可量化——大布局时间步少但路径长（idle 错误多），小布局路径短但旋转频繁。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（评估 pipeline）：①把调度输出的 lattice surgery 指令序列解析为时间片级操作；②用 STIM 对每个原子操作（PPM/PR/measurement）在 d=9、p=10⁻³ 电路级去极化噪声下 Monte Carlo 采样（≥10⁶ 次）得到各错误率，PyMatching 2 解码；③对每个时间片按公式复合 p_layer；④累加所有时间片得总 LER。例子：O3LS 在 7×7 squeezed 布局 vs 10×10 standard 布局跑 adder_28——前者时间步相近但 ancilla 路径短 → P_idle 低 → LER 低（相对大布局降最高 16.9%，相对 SPC 降 43.11%（compact）/44.98%（standard），相对 LAPBC 最高 93.95%）。码距敏感性（d∈[3,5,7,9]）：距离相关的指数级解码抑制对 O3LS 与 baseline 同样适用，故 O3LS 只优化架构因素、收益不随码距衰减。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器的 LER 评估模块（Python 3.10，Intel Core i9-14900K 32 核 + 188GB RAM），输入 = 调度指令序列 + 布局 + 噪声参数，输出 = 总 LER 与每层错误率。该模型也是 O3LS 优化目标——布局搜索的 sweet spot、松散调度、Y-synthesis、EA 映射都以压低 p_layer 为目标。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
