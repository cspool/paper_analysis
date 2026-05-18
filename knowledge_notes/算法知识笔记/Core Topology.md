## Core Topology

术语是什么？

Core Topology 是DMI将UI Navigation Forest压缩为LLM context window友好的层级文本描述的技术。原始UNG/Forest包含5K+控件节点，若全部序列化入prompt将远超合理token预算（即使大context，过长控件列表也会稀释task-relevant信息、增加推理成本）。Core Topology通过分层描述+连续整型ID编码，将完整forest保持在一个可管理的token预算内（Excel ~30K tokens, Word ~15K, PowerPoint ~15K），同时保留按需扩展能力（`further_query`机制）。

从算法pipeline角度拆解术语：

Core Topology的压缩pipeline：

```
输入: Path-Unambiguous Forest (main tree + shared subtrees)
输出: Core Topology文本 (分层描述 + 整型ID映射)

1. 整型ID映射:
   将冗长XPath-like控件ID: "Blue|ControlType.Button|ribbon/Design/FormatBackground/FillColor/"
   替换为连续整型ID: 128

2. 层级结构序列化:
   使用紧凑格式: name(type)(description)_id[children]
   例: "Design(Tab)_12[FormatBackground(Group)_45[SolidFill(RadioButton)_67, ...]]"

3. 分层描述策略:
   - Level 1 (Core): 最常用的功能路径 → 默认包含在prompt中
   - Level 2+ (Extended): 低频或专门化功能 → 按需通过further_query扩展
   - 非叶节点: 保留但标记为navigation-only
   - 叶节点: 可直接作为LLM声明访问的目标

4. Shared subtree引用:
   共享子树不重复序列化，在引用处标注entry_ref_id
   例: "...[ApplyToAll(Button)_132@entry_ref=15]"
```

压缩密度：原始5K+控件 → ~15K-30K tokens → LLM可一次加载并在多次交互轮次中复用。

术语一般如何实现？如何使用？

Core Topology在DMI prompt中的使用：(1) 作为system message或task preamble注入LLM prompt；(2) LLM通过阅读层级描述理解应用的功能组织——类似于压缩的"功能地图"；(3) LLM声明目标时引用整型ID（如 `"ids": [128]`），DMI executor反向映射为XPath-like path；(4) 当LLM需要的功能不在core topology中（提示not found），可通过`further_query`请求扩展对应子树（DMI动态追加到prompt context）；(5) 离线构建时决定哪些分支放core vs 需要query——论文报告Excel未压缩core topology ~30K tokens，人工配置决定branch的core/extended划分。与直接将5K+控件全量注入prompt相比，Core Topology既保持了信息完整性（需要时扩展），又避免了context dilution（默认仅core）。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

