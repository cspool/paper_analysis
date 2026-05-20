## LEGO: Supporting LLM-enhanced Games with One Gaming GPU

- 属于Serving调度的实现是什么？实验比较什么？
  提出headroom-maximizing LLM scheduler，实现LLM推理与游戏渲染在单张消费级GPU上的细粒度共置调度。核心调度设计：(1) 修改llama.cpp的traversal function加入调度逻辑：游戏引擎(UE4)监控rendering task状态变量→渲染完成后启动inference subtask→decode阶段以Transformer layer为调度粒度(~0.4ms/layer)、prefill阶段以self-attention和FFN sublayer为调度粒度(~0.5-1.0ms)；(2) Headroom Prediction：以LLM inference execution window为时间单位（100 APM=600ms含36帧、200 APM=300ms含18帧、300 APM=200ms含12帧），用LR模型以前三个window总rendering headroom预测下一个window总headroom，最大预测误差1.3%、平均0.6%、推理开销1.3ms；(3) Feedback-driven intra-rendering调度：监控rendering subtask的start/completion→当rendering subtask完成且下一subtask未开始时提交fine-grained LLM subtask→利用intra-rendering headroom（平均0.24ms/gap, 总平均1.39ms/frame）；(4) Inter-rendering调度：渲染任务完成→切换coarse-grained LLM subtask（含多个transformer layers）→利用帧间headroom；(5) Safety constraint：T_subtasks ≤ T_minimal（T_minimal为游戏所有rendering task中最小的inter-rendering headroom），保证利用intra-rendering headroom不会导致渲染任务latency violation；(6) Sudden spike处理：每个token生成后用最新workload数据更新预测→检测QoS violation风险→动态调整后续token的layer-skipping策略。实验比较LEGO vs SmallModel（同系列小模型）和LayerSkip（LITE/CALM），在三种游戏（BlackMyth/FFXVI/RDR2）×两种LLM（Llama3-8B/Mistral-7B）×三种APM（100/200/300）共18个场景下，测量99th-percentile FPS和APM。LEGO在所有场景同时满足FPS和APM目标。

- 硬件平台是什么，配置是什么。
  Windows 11, CUDA driver 566.36, CUDA SDK 12.1, DirectX 12.1。Intel i9-13900KF @ 3.00 GHz, Nvidia RTX 4090 (24GB)。所有游戏配置4K高画质60 FPS。

- 开源Serving框架是什么。修改了什么。
  基于llama.cpp (github.com/ggml-org/llama.cpp, commit fc83a9e) 作为LLM inference framework。游戏引擎为Unreal Engine 4，图形库DirectX 12。核心修改：(1) 只将llama.cpp front-end集成到UE4中，其他功能通过dynamic library调用；(2) 修改llama.cpp的traversal function（computation graph creation与traversal分离）加入调度逻辑：游戏引擎监控rendering task状态变量→渲染完成时dispatch inference subtask→decode阶段dispatch transformer layers、prefill阶段dispatch self-attention/FFN sublayers；(3) 在dynamic library中注册新的schedulable traversal function，保证推理执行正确；(4) 集成LR headroom predictor：运行时取前三个execution windows的总headroom预测下一个window。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供官方开源仓库。以BlackMyth + Llama3-8B + 200 APM为例说明调度流程：
  1. 部署：LEGO scheduler集成到UE4游戏引擎中→游戏启动时加载llama.cpp dynamic library→注册schedulable traversal function→离线准备好的adaptor模型权重加载到GPU memory
  2. 运行时：游戏以60 FPS渲染→每16.6ms一帧→scheduler监控每个rendering subtask的start/end状态
  3. LLM action到达（每300ms）：scheduler用LR模型预测接下来300ms window内的总rendering headroom（基于前三个window的headroom历史）→选择跳层策略（如跳4层）→选择合适的adaptor
  4. Prefill阶段：scheduler将self-attention和FFN sublayer作为调度粒度→当rendering subtask完成且下一subtask未开始时→提交一个attention/FFN subtask→执行约0.5-1.0ms→检查下一rendering subtask状态→若已开始则等待→否则继续提交
  5. Decode阶段：每个token生成时→以Transformer layer为粒度（~0.4ms）→利用intra-rendering headroom填充→整帧渲染完成后进入inter-rendering headroom→提交coarse-grained subtask（多个transformer layers）
  6. Safety check：每个提交的LLM subtask满足T_subtask ≤ T_minimal（该游戏最小inter-rendering headroom）→保证不阻塞渲染
  7. Sudden spike处理：生成每个token后用最新workload更新headroom预测→若检测到QoS violation risk→调整后续token的跳层策略
  8. Headroom usage：LEGO相比SmallModel在100/200/300 APM下分别提升25.2%/28.6%/18.8%；相比LayerSkip在200/300 APM下分别提升14.0%/16.2%

