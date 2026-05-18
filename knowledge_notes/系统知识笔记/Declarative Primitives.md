## Declarative Primitives

术语是什么？

Declarative Primitives 是DMI定义的三种LLM声明式操作原语：access、state、observation。这些原语构成LLM Agent与GUI之间的声明式接口契约——LLM仅通过这三类原语表达意图，无需输出细粒度的GUI操作序列（点击坐标、菜单导航、拖拽路径等）。这种设计将GUI操作从LLM的推理空间（imperative action space，需处理每个可见控件的具体定位和操作）提升到语义层面（declarative intention space，仅表达要达成什么状态/读取什么信息）。

从系统架构角度拆解术语：

三类声明式原语的系统级语义：

1. **access原语**：声明访问目标控件的意图。LLM输出目标控件在core topology中的leaf ID（连续整型），shared subtree中的控件需附加entry_ref_id。DMI executor负责：(a) 解析ID为XPath-like path；(b) 在forest中求解唯一导航路径；(c) 在当前窗口fuzzy matching控件；(d) 沿路径点击导航节点；(e) 最终invoke/select目标控件。单次access声明可替代GUI-only中多步"展开菜单→切换tab→展开dropdown→点击选项"的序列。

2. **state原语**：声明设置控件状态的意图。LLM调用 `set_scrollbar_pos(percentage)`、`select_lines(start, end)`、`select_controls([id1, id2, ...])` 等结构化接口，而非输出拖拽坐标或多次点击序列。DMI executor利用UIA ScrollPattern/TextPattern/SelectionPattern将状态设置转化为pattern调用。

3. **observation原语**：声明观测控件内容的意图。LLM调用 `get_texts()` 获取结构化文本而非依赖像素级OCR+额外的observe-act循环。DMI有两种模式：passive mode在每次LLM调用前自动采集DataItem截断文本；active mode按需返回完整文本。

术语一般如何实现？如何使用？

声明式原语是接口抽象，具体实现：(a) 原语定义为结构化JSON schema——每个LLM action以JSON格式输出（如`{"action": "access", "ids": [128, 132], "entry_ref_id": 15}`）；(b) DMI executor解析JSON→调用对应的UIA pattern或导航路径求解器→返回结构化结果（成功/失败+available选项/控件树）；(c) 原语之间可组合：LLM可在一个action中混合access+state（先导航到控件再设置状态）；(d) 原语失败时返回结构化错误信息（控件not found/ambiguous match/pattern not supported），供LLM在下一轮调整声明策略。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents
