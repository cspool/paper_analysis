## Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：在 wafer-scale multi-chiplet GPU 的 Global Command Processor 上运行的两个运行时算法：(1) **Task Allocation Algorithm (Alg. 1)**——将 MoE kernel 计算按 expert 拆分为 per-die 子 kernel，基于 expert placement 信息和 cost model（考虑 DRAM access、computation、D2D communication）将子任务分配到各 die；(2) **Data-Driven Predictor**——利用 cross-token heatmap 预测下一 token 的热门 expert，为 hardware-managed HBM 提供 duplication 指导（cp_en bits），在 kernel 执行期间自动将远程热门 expert 缓存到本地 HBM。
  - 实验比较：(1) Allo Only vs Base vs EP 的 throughput 和 hop count（Task Allocation 的独立效果）；(2) Pred Only vs Base vs EP 的 throughput 和 hop count（Predictor 的独立效果）；(3) Allo+Pred 的组合效果；(4) DRAM access breakdown（local reads / remote reads / local writes）展示两种策略如何将远程读转化为本地读；(5) Host CPU-based 实现 vs GPU CP-based 实现的 overhead 对比（Dojo vs Dojo-Enhanced）。

- 后端平台是什么，配置是什么。
  - **模拟平台**：自研 Python 事件驱动 multi-chiplet GPU simulator（开源：https://github.com/zhongkaiyu/waferscale_gpu_moe_sim）。
  - **硬件配置**（两种 chiplet 拓扑）：
    - **Tesla Dojo**：5×5 2D mesh（25 dies），每 die H100-like（1000 TFLOPS FP16, 80GB HBM, 3.35 TB/s HBM BW, 1.7 TB/s D2D BW）。LLC: 64 MB, 100ns hit latency。D2D: 200ns link latency, XY routing。
    - **TSMC SoW**：8×3 2D mesh（24 dies），每 die 相同 H100-like 配置。
    - **Dojo-Enhanced**：5×5 mesh，每 die B300-like（4500 TFLOPS FP16, 180GB HBM, 8 TB/s BW, 2 TB/s D2D BW）。
  - **模型**：DeepSeek V3 (671B, 256 experts), Kimi K2 (1000B, 384 experts), Llama4 Maverick (402B, 128 experts), Qwen3-235B (235B, 128 experts)。
  - **workload**：real traces from MMLU, MMLU Pro (Chinese), ChineseSimpleQA, LiveCodeBench (>24,000 requests per model)。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：自研 Python multi-chiplet GPU simulator（`main_ae.py` 入口），模拟 wafer-scale GPU 上 MoE decode 阶段的 kernel 执行。
  - 修改/新增的核心算法：
    1. **Task Allocation Algorithm (Alg. 1)**：
       - 输入：`expert_reqs_dict`（每个 expert 的请求数）、`expert_die_map`（Expert Distribution Table 提供的每个 expert 所在 die 信息）
       - 候选机制 (Candidate Mechanism)：对每个 expert，候选 die 列表 = 存有该 expert 的 die + 距离为 1 的邻居 die，按负载排序后限制为 `max_split_num`（由 expert 请求数决定）
       - 块粒度分配 (Block-Granularity Distribution)：以 block size = 50 为单位分配请求，每 block 用 cost model（DRAM access + compute + D2D communication）选择最优 die
       - 合并：将分配到同 die 的任务合并为最终 allocation plan
    2. **Data-Driven Predictor**：
       - 输入：当前 MoE kernel 中各 expert 被选择的情况 + cross-token heatmap（离线 pre-computed）
       - 从 heatmap 中定位当前被选择的 expert 对应行（①），每行取 top-n 下一 token 最可能的 expert（②）→ 预测结果为这些 expert 的并集（③）
       - 输出：每个 die 的 cp_en bits（标记哪些 expert 应被本地缓存），写入 PDU prediction table
    3. **Cost Model**：考虑 DRAM access latency（local 300ns, remote 多跳 D2D + remote DRAM）、computation time（基于 TFLOPS）、D2D communication（bandwidth + hop count contention）
  - 实验 metric：decode 阶段 MoE layer throughput（tokens/s），hop count（所有 cross-die 通信的 Manhattan 距离之和），DRAM access breakdown。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：完整代码仓库 https://github.com/zhongkaiyu/waferscale_gpu_moe_sim（Apache-2.0），DOI: 10.5281/zenodo.19617713。提供 `main_ae.py` 入口脚本自动下载 traces → 运行实验 → 生成 CSV → 生成 Figure 12。
  - **评估原理与全流程（以 DeepSeek V3 on TSMC SoW 8×3，batch size 16384 为例）**：
    1. **Trace 加载**：从 HuggingFace 下载 DeepSeek V3 的 expert selection traces（JSON 格式，每层每 token 的 top-k expert IDs）。TSMC SoW 8×3 拓扑初始化——24 个 die 对象，每个含 HBM/LLC/compute/D2D link model。Expert 按 EP-like 均匀分配到 24 个 die。
    2. **Batch 构建**：按 MMLU → MMLU-Pro (CH) → ChineseSimpleQA → LiveCodeBench 顺序填充请求直到达到 target batch size 16384。从 traces 中提取对应请求的 expert selection。
    3. **每层 MoE kernel 执行（decode stage）**：
       - **Global CP 阶段**：(a) 统计当前 batch 中每个 expert 的请求数 → expert_reqs_dict；(b) 运行 Task Allocation Algorithm——对每个 expert（按请求数升序），生成候选 die 列表 = 存有该 expert 的 die + 邻居 die → cost model 评估每个候选 → block size 50 贪心分配 → 合并生成分配计划；(c) 运行 Predictor——从 cross-token heatmap 查当前 expert selection 的对应行 → 每行取 top-n → 预测下一 token 热门 expert → 生成 cp_en bits。
       - **执行阶段**：Global CP 通过 D2D 网络发送子 kernel + prediction 信息到各 Local CP。Each Local CP 分配任务到 SM。SM 请求数据时：(i) PDU 检查 is_local → 若已缓存，ATU 翻译地址从本地读取；(ii) 若远程且未缓存 → D2D XY routing 多跳读取 → 返回时 PDU 检查 cp_en → 若需缓存，写入本地 HBM + 更新 ATU。
    4. **性能统计**：(a) 每层执行时间 = max(all die completion time)，汇总所有 MoE 层得到总执行时间 → throughput = batch_tokens / total_time；(b) Hop count = 所有 D2D 请求的 Manhattan 距离之和；(c) DRAM access 分类统计 local read / remote read / local write。
    5. **关键结果**：Allo+Pred 实现 7.5× throughput on TSMC（vs 6.0× on Dojo，因 TSMC 矩形布局 die 间距更大故 baseline 更差）。Hop count 降低 >213×。Allo Only 的 hop 降低 142× 证明 allocation 已将多数请求分配到本地 die；Pred Only 额外将 remote DRAM reads 转换为 local reads。
