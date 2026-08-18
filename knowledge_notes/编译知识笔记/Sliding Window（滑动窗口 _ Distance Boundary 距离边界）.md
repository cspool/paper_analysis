## Sliding Window（滑动窗口 / Distance Boundary 距离边界）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MANATEE 的滑动窗口是投机页着色的核心机制：窗口尺寸 = 推测的功率开启周期 T_on；窗口从冲突点双向（反向+正向）扩展，覆盖"断电可能发生在冲突点之前或之后"两种情况。窗口内（可能同一功率周期共存）的页面必须不同色；窗口外的页面视为不会共存、其颜色可被回收复用（steal）。
- 距离边界（distance boundary）由 T_on 转换而来：编译器沿 CFG 各路径从冲突点正反向累计指令周期，当累计值 ≥ T_on 时设边界；考虑漏电流等波动，加 20% 安全裕量把边界放得更长。即使实际功率周期超过边界，正确性仍由 page manager 兜底，只是可能增加页 fault。
从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：冲突页 Pg4 着色时无空余颜色 → 反向 DFS 找窗口外颜色 {red, orange}、正向 DFS 找 {orange, green} → 并集 {orange, red, green} → 优先选两个窗口交集颜色 orange → 窃取 orange 复用给 Pg4。若正反向窗口无交集，则优先反向窗口（代表断电前状态，更保守）。滑动窗口使 MANATEE 避免传统着色的永久 spill，允许页在运行期动态竞争 SPM frame。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译器对 CFG 做双向深度优先搜索（reverse/forward DFS），滑动窗口尺寸为 T_on 换算的 MCU 周期；评估上对 12 个 benchmark 用 thermal trace 验证：滑动窗口把页 miss rate 从 2.05%（无窗口）降到 0.99%（~50% 降低）。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
