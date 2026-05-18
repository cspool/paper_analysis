## LEGO: Supporting LLM-enhanced Games with One Gaming GPU

- baseline方法是什么？
  Baseline有三种：(1) SmallModel：用同系列小模型（Llama3-3B替换8B、Mistral-4B替换7B）降低推理开销，运行时按游戏平均rendering headroom将LLM inference切为等大小subtasks，渲染完成后提交一个推理subtask；(2) LayerSkip（LITE/CALM）：基于token-level confidence threshold做per-token early exit/layer-skipping决策，跳层策略确定后用与SmallModel相同调度方式。LITE对每层定义预定义confidence threshold→当某token在某层confidence超过threshold→跳过剩余层输出该token；(3) PilotFish式time-division GPU sharing：在渲染任务结束后利用帧间idle time dispatch LLM subtask，但不利用intra-rendering headroom。

  全栈执行例子（以LITE + Llama3-8B + BlackMyth + 200 APM on RTX 4090为例）：
  - 算法层：LITE对Llama3-8B fine-tune→每层定义confidence threshold。推理时每个decode token经过各transformer layer→计算confidence score→若在某层超过threshold→early exit跳过剩余层→head映射到vocabulary→输出token。不同token退出深度不同，平均跳层数无per-request保证。
  - 系统框架/Serving层：LITE + PilotFish scheduling。游戏以60 FPS渲染→每16.6ms一帧。LLM action到达（每300ms, 200 APM）→PilotFish在每帧渲染完成后dispatch等大小LLM subtask→仅利用inter-rendering headroom。scheduler不进行headroom预测，不拆分intra-rendering subtask。
  - 编译框架层：论文未明确说明（llama.cpp默认CUDA编译路径）。
  - kernel调度层：论文未明确说明（llama.cpp默认CUDA kernel，无定制kernel）。
  - 硬件架构层：Nvidia RTX 4090，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LEGO，algorithm-system co-design：算法侧用resource-oriented layer-skipping adaptor做知识蒸馏补偿；系统侧用headroom-maximizing LLM scheduler做细粒度共置调度。

  **缺陷1：SmallModel固定用小模型导致精度永久下降，即使GPU有空余headroom也无法恢复精度**
  → LEGO保留原始大模型主体权重不变，只在资源不足时按需跳过特定层并用adaptor补偿。当GPU headroom充足时（如100 APM），仅需跳≤5层即可满足90% case，精度接近原始模型。相比之下SmallModel的精度drop固定为平均20.41%（MMLU/ARC-C/SQuAD-2.0）。

  **缺陷2：LITE/CALM的token-level confidence-based跳层无法为每个请求提供资源预算保证，导致SLO violation**
  → LEGO反转跳层决策逻辑：先由资源预算决定跳层数量（scheduler预测headroom→选择层数），再在该约束下通过adaptor做知识蒸馏补偿。LITE即使在平均时间对齐latency target时仍有47.1%推理超时；LEGO确保每个action请求在execution window内完成，99th-percentile APM全部达到目标。

  **缺陷3：LITE-S强行加入SLO约束需要提前跳层，跳过了其自身机制认为重要的层，导致27.2% accuracy drop**
  → LEGO的adaptor在离线阶段通过similarity heatmap选择相似度最高的连续层段做蒸馏，而非逐token动态决定。跳层时用训练好的FFN adaptor替代跳过的层段，在跳12层时相比LITE减少86.3% accuracy loss。

  **缺陷4：PilotFish只利用inter-rendering headroom，高APM下GPU空闲不足导致大量跳层甚至FPS/APM violation**
  → LEGO scheduler同时利用inter-rendering（帧间）和intra-rendering（帧内）headroom。测得intra-rendering headroom平均0.24ms/gap（总平均1.39ms/frame, 最大3.1ms）。将LLM subtask粒度降到layer级（decode ~0.4ms）和sublayer级（prefill: attention ~0.5ms, FFN ~1.0ms）以填充这些极短的空隙。headroom usage提升最高28.6%。

  **缺陷5：逐帧headroom预测误差大，naive时间序列模型预测误差>3%且开销高（ARIMA~1s, SVM>50s)**
  → LEGO采用以execution window为单位的LR预测（前三个window总headroom→预测下一个window），最大预测误差仅1.3%、平均0.6%。因为window跨12-36帧，单帧波动被平滑。LR推理开销仅1.3ms (3-input) 或 0.9ms (runtime fit)。

  论文方法全栈执行例子（以LEGO + Llama3-8B + BlackMyth + 200 APM on RTX 4090为例）：
  - 算法层：离线训练adaptor。构建layer similarity heatmap→跳4层选L25-L29、跳8层选L23-L31等→每种跳层配置训练FFN adaptor（MSE loss）。在线推理：scheduler决定跳N层→替换对应transformer层段为adaptor→adaptor输入f_k输出f_{k+n}近似原表示→剩余层正常前向→输出token。
  - 系统框架/Serving层（核心创新）：llama.cpp front-end集成到UE4→修改traversal function加入调度→scheduler用LR基于前3个execution window总headroom预测下个window→选择跳层策略→prefill以attention/FFN sublayer为粒度利用intra-rendering gap→decode以transformer layer为粒度→rendering subtask完成时提交fine-grained LLM subtask（T_subtask ≤ T_minimal safety check）→整帧完成后切换coarse-grained subtask利用inter-rendering gap→每token生成后更新预测→检测QoS violation则动态调整跳层。
  - 编译框架层：论文未明确说明（llama.cpp默认CUDA编译路径，修改限于traversal调度逻辑）。
  - kernel调度层：论文未明确说明（使用llama.cpp默认CUDA kernel，无新定制kernel。调度发生在traversal layer，非kernel level）。
  - 硬件架构层：Nvidia RTX 4090消费级GPU，无定制硬件。论文强调LEGO不依赖RTX 4090特殊特性，可部署到其他gaming GPU。
