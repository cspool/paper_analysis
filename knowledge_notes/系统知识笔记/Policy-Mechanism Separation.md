## Policy-Mechanism Separation

术语是什么？

Policy-Mechanism Separation 是操作系统和系统架构中的经典设计原则：将"做什么"（policy）与"怎么做"（mechanism）解耦。在DMI论文中，该原则被应用于LLM Agent与GUI交互：LLM承担policy角色（语义理解、任务分解、意图声明），DMI承担mechanism角色（控件定位、导航路径求解、交互执行）。这一设计的核心insight是LLM擅长语义推理和高层规划，但不擅长需要精确视觉定位和确定性导航的低层GUI操作；将mechanism编码为确定性算法（图搜索、UIA tree匹配），而非让LLM通过prompt推理执行路径，可从根源消除大量mechanism-level失败。

从系统架构角度拆解术语：

Policy-Mechanism Separation在DMI中的具体体现：

1. **Policy层（LLM）**：接收task + core topology + 声明式API说明→输出结构化JSON声明意图（如 `{"action": "access", "ids": [128, 132], "entry_ref_id": 15}` 或 `{"action": "state", "type": "set_scrollbar_pos", "value": 80}`）。LLM不输出点击坐标、拖拽路径、菜单展开序列等mechanism细节。

2. **Mechanism层（DMI）**：接收LLM的声明式意图→(a) 解析控件ID为UIA control path（XPath-like: `primary_id|control_type|ancestor_path`）；(b) 在path-unambiguous forest中求解root-to-target唯一路径；(c) fuzzy matching当前窗口实际控件（处理名称变化）；(d) 执行导航点击序列；(e) 通过UIA control patterns执行状态设置；(f) 返回结构化结果/错误反馈。

3. **消融实验验证**：为baseline注入DMI static navigation knowledge（文本或JSON描述），但不启用声明式接口——LLM仍需通过命令式接口利用该知识。结果SR从44.4%降到42.0%，步骤数未减少。这证明分离的**接口形式本身**（而非仅知识内容）是performance gain的来源。

术语一般如何实现？如何使用？

Policy-Mechanism Separation在系统设计中通用模式：(1) 定义清晰的interface contract——DMI用access/state/observation三类原语；(2) mechanism通过离线建模获取足够信息以实现确定性执行——DMI的UNG + path-unambiguous forest；(3) policy通过压缩的模型（core topology）获得足够信息以做出语义决策而不被mechanism细节淹没；(4) mechanism不可行时fallback到policy的原始路径——DMI保留GUI slow-path。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---
