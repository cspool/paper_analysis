## 干球温度与湿球温度（Dry-Bulb Temperature DBT / Wet-Bulb Temperature WBT）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 干球温度（DBT）是标准温度计（遮蔽直射辐射与水分）测得的空气实际温度，即日常天气报告中的"气温"，代表空气的显热；湿球温度（WBT）是在湿布包裹温度计球泡、空气绝热蒸发冷却条件下能达到的最低温度，代表空气的蒸发潜力。两者通过相对湿度关联：DBT 30°C、RH 40% 时 WBT≈20.5°C，而 RH 90% 时 WBT 升至 28.6°C。数据中心的两种散热路径分别依赖二者：空气冷却（空气侧 economizer 直通风、风冷 chiller 屋顶风机散热器）把热量直接排入环境空气，能力由 DBT 决定；水冷却（蒸发冷却塔）靠水分蒸发带走热量，极限温度是 WBT——WBT 越高蒸发潜力越低、冷却能力越差。经验法则：DBT 升 1°C 只推高 WBT 约 0.25°C（除非湿度同步上升），因此 WBT 极值的绝对幅度通常小于 DBT 极值，但高湿地区（如 Council Bluffs、Dalles）的湿球风险更关键。
- 该术语从硬件架构角度拆解：DBT/WBT 是冷却硬件设计的外部约束变量，决定"该配多少冷却容量"。运行流程：服务器产热 → 冷却系统（economizer/chiller/冷却塔）把热排入环境 → 环境 DBT/WBT 越高、排热越困难 → 冷通道温度（CAT）越难维持。Prometheus 按站点环境敏感性分流：DBT 敏感站点（Dublin、London、Phoenix，主要用风冷）看 DBT 50 年回返温度；WBT 敏感站点（Council Bluffs、Dalles，蒸发冷却）看 WBT 50 年回返温度。Table II 示例：London DBT 50 年回返 ASHRAE 37.7°C → Prometheus 2044 预报 41.2°C；Dalles WBT 24.5°C → 27.1°C。Fig.12 显示随 WBT 升高蒸发冷却容量快速退化（WBT>30°C 后 chiller 饱和），迫使加装模块化冷却单元或削减负荷。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：WBT 可查湿空气焓湿图或用 psychrometric 公式从 DBT、RH（或露点）计算；气象站（如 London St. James's Park, WMO:037720）提供 DBT 观测，WBT 常需推导。CMIP6 模拟只直接提供 DBT 与 RH，且网格粗、时间错位，无法用标准公式可靠算 WBT——这正是 Prometheus 用两级 ML 集成（SVM+RF→NN）从 DBT/RH 学 WBT 的原因（RMSE 1.71→0.67°C）。
  - 使用：设计阶段按站点 DBT/WBT 的 N 年回返温度选冷却技术（纯蒸发冷却 vs 加装 chiller）与容量；运营阶段用短期 WBT/DBT 预报触发负荷削减/迁移。术语本身属气象/暖通领域，跨论文可复用于任何依赖环境温度的数据中心散热设计。

涉及论文标题：
- Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure
