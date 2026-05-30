## Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing

- baseline方法是什么？
  - Baseline 是现有的简单预测方法（如 moving average——直接使用历史数据均值作为预测结果），以及负载均衡的辅助 loss 方法（如 load balancing loss 加正则项、capacity factor 限制 experts token 处理上限、expert-based routing / hash-based routing）。Baseline 的核心问题：(a) 无法区分 MoE 训练过程中 expert 负载的 transient state（早期波动阶段）和 stable state（后期稳定阶段），导致在波动阶段预测无效，在稳定阶段预测精度不足；(b) 现有方法（FlexMoE/Prophet）使用启发式或 moving average 进行负载预测，在不平衡负载持续存在且状态转换时精度受限。
  - 全栈执行例子（Baseline: Moving Average 预测 + Load Balancing Loss, GPT-3 350M training, 4×A800）：
    - **算法层**：Gating network 对每个 token 执行 Softmax(W_gate·x) → Top-K routing 选择激活 experts。Model training 阶段加入 auxiliary load balancing loss（expert activation entropy 之和）迫使 gating 网络均衡分配 token。Moving average 预测直接取历史 t 轮的 expert 负载均值作为下一轮预测——在 transient state（前约 5,000 iterations）中因负载剧烈波动而预测不准；在 stable state 中因未建模时序相关性而精度受限。
    - **系统框架层**：论文未明确说明 training framework 细节。FlexMoE 使用启发式算法基于负载动态优化 expert placement；Prophet 利用 temporal locality 进行 layer-wise fine-grained 资源调度——但两者依赖的负载预测在 transient state 下不可靠。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明——预测算法在 CPU 上执行，不涉及 GPU kernel 修改。
    - **硬件架构层**：4×A800 GPU。在 MoE training 中 experts 可能因负载不均导致 GPU 资源浪费——hot expert 所在 GPU 计算拥塞而 cold expert 所在 GPU 资源闲置。Baseline 在全量 load balancing loss 下虽均衡了 expert 负载但可能影响模型精度（过度干预 gate network）。
  - Baseline 核心缺陷根因：现有方法无法准确判断 expert 负载何时从 transient 转换为 stable state，也无法在两种状态下提供高精度的负载预测，导致基于预测的 expert placement 和 resource allocation 决策不可靠。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：通过大量 MoE 模型训练实验系统性追踪和分析 expert 负载的时序变化，定义 transient state 和 stable state，并部署三种经典预测算法（LSTM、ARIMA、SW Avg）在不同状态下实现高精度 expert 负载预测。
  - 全栈执行例子（GPT-3 350M, 4×A800）：
    - **算法层（解决"无法区分负载状态"和"预测精度低"的缺陷）**：
      - 状态定义：通过计算各 expert 负载在不同滑动窗口（w=10,100,200）下的方差和极差，发现 expert 负载依次经历 transient state（早期波动、无规律变化）和 stable state（负载稳定、呈现 temporal locality——相邻迭代间负载变化微小）。以 GPT-3 125M 为例，约 5,000 iterations 后进入 stable state。
      - LSTM-based 预测：输入为所有 MoE 层所有 expert 的历史负载比例序列 [n_1^1,...,n_e^m]，输出为未来 k 步负载比例值；两次独立训练数据分别作为 train/test set。GPT-3 350M stable state 误差率约 <10%（每 1,000 iterations 预测粒度）。
      - ARIMA-based 预测：对每个 expert 负载时序做平稳性/季节性检验选取 ARIMA(p,d,q) 参数，实验中 ARIMA(5,1,5) 取得低于 LSTM 的误差率。GPT-3 350M stable state 误差率约 1.4%。
      - SW Avg-based 预测：算术平均历史多轮数据直接作为预测值，通过 k 轮滑动计算预测未来。该算法在三种方法中表现最佳：GPT-3 125M stable state 误差率约 0.25%，GPT-3 350M stable state 误差率约 1.3%（1,000-step）和 1.7%（2,000-step）。方法计算极其简单，硬件友好。
      - 对比 baseline：Baseline 的 moving average 不做状态区分且不建模时序依赖；论文方法通过 LSTM/ARIMA 捕捉时序模式提升精度，通过 SW Avg 在极低计算成本下获得最佳精度，并根据状态区分指导资源分配策略——stable state 下可基于预测精细分配，transient state 下需预留充足资源应对波动。
    - **系统框架层**：论文未明确说明具体框架修改。预测结果可对接 FlexMoE（启发式 expert placement 算法）或 Prophet（fine-grained layer-wise 资源调度），在 stable state 下提供准确的 per-expert 负载信息作为 placement/scheduling 决策依据。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明——预测算法为 CPU 端时序分析任务，不修改 GPU kernel。
    - **硬件架构层**：同一 A800/4090 GPU 平台。核心变化：通过高精度负载预测使基于负载的 expert placement（将 hot expert 部署到更多 GPU、cold expert 合并部署）成为可行——在 stable state 下约 1.3% 误差率意味着可精确预知 GPU 资源需求、提前规划 expert-to-GPU 映射，减少 resource underutilization 和 load imbalance 导致的训练效率损失。在 transient state 下由于波动不可预测，保留充足 GPU 资源以应对负载突发。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"无法区分负载状态"**：通过大规模实验统计分析定义 transient/stable state 的量化特征（方差和极差的时序趋势），并指出 shallow MoE layer 的波动幅度大于 deep layers（spatial dimension 观察）。为后续 resource allocation 提供分阶段策略基础。
    2. **针对"预测精度低"**：部署三种预测算法在两种状态下量化评估——ARIMA 优于 LSTM，SW Avg 在极简计算下反而最佳（因为 stable state 下负载具有强 temporal locality，简单平均即高精度），在 GPT-3 350M stable state 下仅 1.3% 误差率。
    3. **指导价值**：高精度负载预测结果为 downstream 的 expert placement 和 resource allocation 提供可靠输入——在 stable state 下可动态按 expert 负载分配 GPU 资源（hot expert 占更多资源、cold expert 占更少资源），最大化训练效率的同时最小化资源消耗。论文在此篇中仅提供预测能力，并声明后续 work 将设计具体的 expert placement 方案。
