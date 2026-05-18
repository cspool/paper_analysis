## Declarative Model Interface (DMI)

术语是什么？

Declarative Model Interface (DMI) 是一种面向LLM Computer-Use Agent的OS声明式接口中间层。其核心思想是policy-mechanism separation：将LLM Agent与GUI应用的交互从命令式（LLM规划和输出完整的点击、拖拽、键盘等操作序列）重构为声明式（LLM仅声明语义意图——要访问什么控件、设置什么状态、观察什么内容——由DMI负责导航路径求解和交互执行）。DMI不要求修改应用源码，也不依赖应用私有API，而是基于OS accessibility框架（Windows UI Automation）从已有GUI中抽取控件和导航关系，将GUI抽象为access（声明访问目标控件ID）、state（声明设置控件状态如滚动条位置/选中范围）、observation（声明读取文本）三类原语。DMI原型超过18K行Python，基于pywinauto + Windows UI Automation实现，集成到Microsoft UFO framework。

从系统架构角度拆解术语：

DMI在Computer-Use Agent系统中的执行流程：

1. **离线建模阶段**：(a) 通过Windows UI Automation从目标应用（如Word、Excel、PowerPoint）抽取完整的accessibility tree；(b) 使用DFS + differential capture构建UI Navigation Graph (UNG)——采集当前控件树，点击候选控件后再次采集，新出现的控件构成导航边；(c) 将带环和merge node的图转为path-unambiguous forest（包含main tree和shared subtrees）；(d) 压缩为LLM友好的core topology层级文本描述，将冗长XPath-like控件ID替换为连续整型ID。

2. **在线执行阶段**：(a) LLM接收task + core topology层级描述 + 声明式API说明的prompt；(b) LLM生成声明式action JSON（声明目标控件leaf ID或`set_scrollbar_pos`/`select_lines`等state原语调用）；(c) DMI executor解析控件ID→求解唯一导航路径（确定性图搜索而非LLM规划）→fuzzy matching当前窗口控件→按照OK>Close>Cancel优先级关闭浮窗→沿路径点击导航控件；(d) 到达目标控件后通过UIA control patterns（ScrollPattern/TextPattern/SelectionPattern/InvokePattern）执行状态设置或数据采集；(e) 每次LLM调用前通过passive mode采集DataItem截断结构化文本，`get_texts()` active mode按需返回完整文本。GUI imperative path保留为slow-path fallback用于非标准控件场景。

术语一般如何实现？如何使用？

DMI使用举例（以PowerPoint "将所有幻灯片背景设为蓝色"为例）：
- **GUI-only baseline**：LLM必须规划并执行 click Design→click Format Background→select Solid fill→click Fill Color→select Blue→click Apply to All，每步均需LLM推理，任一步失败导致级联失败。
- **DMI模式**：LLM从core topology中识别`Blue`和`Apply to All`对应leaf ID→声明access请求（含leaf ID和shared subtree的entry_ref_id）→DMI executor确定性导航到目标并执行。GPT-5 medium下SR从44.4%→74.1%（+29.6%），步数从8.16→4.61（-43.5%），时间从392s→239s（-39%）。

开源：https://github.com/dmi-interface/DMI（MIT License），集成到Microsoft UFO Windows agent framework，提供Office Word/Excel/PowerPoint预构建core topology。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---
