## TCO（Total Cost of Ownership，数据中心总拥有成本）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TCO（总拥有成本）是数据中心全生命周期成本的总和，分为 CapEx（资本支出）与 OpEx（运营支出）两部分。本文（Rearchitecting the Datacenter Lifecycle for AI，ISCA'26，UT Austin + Microsoft Azure）的 TCO 模型（表 III）把 CapEx 拆为 CapEx_F（设施建筑）+ CapEx_Pow（电源供电 $7.0/W）+ CapEx_Cool（冷却 $2.5/W）+ CapEx_Net（网络 $2000/server）+ CapEx_IT（IT 硬件 $375k/server，DGX H100 参考）；OpEx 拆为 OpEx_energy（按 IT 负载 × PUE × 电价 $20–40/MWh）+ OpEx_M&R（维护 $5000/server）+ OpEx_network（$600/server）+ OpEx_other（软件许可 $200/server）。设施/网络/IT 分别按 15–30、7–10、3–5 年直线或余额递减折旧，CapEx 按每交付 kW 归一化以便跨设计比较。论文的核心创新不是 TCO 公式本身（经典框架早已有之），而是把 TCO 作为统一优化目标：把"模型演进 × 硬件 roadmap × 基础设施决策 × 运营软件优化"全部折算成 15 年（60 季度）的 TCO，用蒙特卡洛枚举最优刷新策略，实现跨 build/IT provisioning/operation 三阶段的联合优化（相对传统割裂式管理最高降 40%）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TCO 作为容量规划目标函数的运转流程（以 10MW 数据中心为例）：
```
输入: 硬件 roadmap(P100→B200 的 FLOPS/带宽/TDP/价格), 模型趋势(Epoch AI),
      需求(DynamoLLM trace, 100K RPS, 15%/年), PUE, 电价, 折旧期, 刷新策略
每季度 t:
  1. roofline 模型预测每个 (模型, GPU) 的 TTFT/TBT
  2. 在 SLO(TTFT≤400ms, TBT≤100ms) 下增压求 goodput(最大可持续 RPS)
  3. 按需求 RPS(t) / goodput 求最小 GPU 数 → 采购/退役计划
  4. CapEx = Σ 新增硬件+电源+冷却+网络(按折旧摊销)
     OpEx  = IT负载×PUE×电价 + 维护 + 网络 + 软件
  5. TCO(t) = CapEx(t) + OpEx(t) → 累加得生命周期 TCO
蒙特卡洛(10,000 trials) 遍历各代 0–10 年生命周期策略 → TCO 分布与 95% CI → 取最优
```
该流程把"某季度买不买新 GPU、旧 GPU 是否退役"变成可量化的 TCO 比较：如 2027 年跳过 B100/B200、延长 H100/H200 寿命（图 13），或在 2024 年 DeepSeek V3 发布时因需求跳升触发 H200 大刷新（server 数从 50 台 2015 年 P100 增至 25K，年 TCO 从 $0.2M 升至 $0.3B）。Sobol-style 敏感性分析给出各策略的 TCO 方差来源（表 X：compute density ±35%、interconnect ±55% 容差内 BOP 仍严格最优）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为开源框架 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass，MIT，Python 包 dc_tco/，模块 config/hardware/models/performance/demand/tco/policies/simulation/scenarios/rental/plotting/cli.py，YAML 配置 + notebooks + tests）。使用：`pip install -e .` → `dc-tco run --config configs/default.yaml --policy baseline` / `dc-tco sweep` / `dc-tco monte-carlo --trials 10000`；Python API 为 `load_config()` + `run_simulation(cfg)` 返回 `result.total_tco`。所有输入（硬件规格/roadmap、workload 特性、基础设施假设、经济参数）可参数化，支持自定义刷新策略、新加速器代际与结构性突变场景（需求冲击/模型收缩/硬件能力跳变/价格冲击）。局限（论文自述）：成本输入依赖公开数据而非专有部署数据，绝对 TCO 因供应商谈判价格而异，价值在于相对策略比较与跨阶段协同的定性洞见。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
