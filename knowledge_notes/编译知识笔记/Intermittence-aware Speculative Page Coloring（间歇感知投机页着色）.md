## Intermittence-aware Speculative Page Coloring（间歇感知投机页着色）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 MANATEE 的核心创新：针对 EHS 频繁断电的特性，只要求冲突页在"推测的功率开启周期（T_on 滑动窗口）"内分配不同颜色，允许被断电隔开的页共享同一 SPM frame（投机它们不会共存）。由于冲突解决被限制在极短窗口，SPM frame 压力大幅降低、页 miss 减少。
- 投机必然有风险（misspeculation）：若实际功率周期长于预期（环境能量比预期强），两页同周期访问同一 frame 会争用，可能破坏正确性（页被错误预测死亡而占两个 frame，或两页误共享一帧）。MANATEE 用运行时 page manager（Buffer Table 判定 + 加密驱逐/解密载入）兜底保证正确性，实测页 miss rate ≈1%。
从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：①干扰图着色遇冲突（颜色不足）→ ②从冲突点反向+正向 DFS 遍历 CFG 路径累计指令周期（静态能量/时序成本模型），以 T_on 换算的 MCU 周期为边界（加 20% 安全裕量）→ ③C_available = C_total − C_accessed 即窗口外可回收颜色，优先取正反向窗口交集颜色、否则优先反向窗口（反向代表断电前的执行状态）→ ④窃取该颜色复用 → ⑤循环/I/O 页（迭代次数未知）保留专用颜色、不参与回收 → ⑥为每条 load/store 生成 (页号, 颜色) hint 与 metadata。结果：页 miss 从 2.05% 降到 0.99%，相对 Memory Coloring 平均快 ~12%，相对 Mapi-Pro 快 2–3×。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：纯编译器 + 运行时软件方案（LLVM 插桩 + TI MSP430 GCC 链接），无硬件修改；功率周期 T_on = E_buf/(P_device − P_input)，P_device = V_dd·I_leak + C_msp·V_dd²·f（参数取自设备手册 [16-20,47]）。误估敏感性：T_on 误估 100%（假设频繁断电而实际不断电）最多 +16% 性能开销。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
