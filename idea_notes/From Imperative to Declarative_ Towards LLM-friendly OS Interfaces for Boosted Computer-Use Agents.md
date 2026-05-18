## From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

- baseline方法是什么？
  Baseline是Microsoft UFO-2的GUI-only模式（UFO2-as + action sequence）。该baseline完全通过GUI与OS应用交互：LLM必须观察当前屏幕/控件树，规划并输出可见控件上的细粒度操作序列——点击菜单项、切换tab、展开dropdown、拖动滚动条、观察反馈再继续。这是一个命令式（imperative）交互范式：LLM承担从高层task规划到底层UI导航和操作的完整责任。

  全栈执行例子（以PowerPoint "将所有幻灯片背景设为蓝色" + UFO-2 GUI-only + GPT-5 medium + Windows为例）：
  - 算法/Agent层：LLM接收task→规划操作序列→每步输出具体动作（click Design tab、click Format Background、select Solid fill、click Fill Color、select Blue、click Apply to All）。每步需一次LLM推理调用，6+步长操作链易级联失败。成功仅44.4%，mechanism-level失败占53.3%。
  - 系统框架/Serving层：UFO-2 agent framework管理LLM调用轮次——每轮等待LLM输出→通过UIA/屏幕坐标执行动作→观察结果→构造下一步prompt。即使注册UIA event handler暴露完整control tree，LLM仍需将控件信息按alphabet labels嵌入prompt后，输出可见控件上的操作序列（而非声明意图）。平均步数8.16，时间392s。
  - 编译框架层：论文未明确说明（Agent级别不涉及编译框架）。
  - kernel调度层：论文未明确说明（不涉及GPU kernel）。
  - 硬件架构层：论文未明确说明（Windows通用PC，无定制硬件）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Declarative Model Interface (DMI)，核心是policy-mechanism separation：LLM只负责"语义上要做什么"的policy声明，DMI负责"如何导航到控件并完成交互"的mechanism执行。DMI分三步：(1) 离线构建UI Navigation Graph→转path-unambiguous forest；(2) 压缩为LLM友好core topology层级文本（将冗长XPath-like控件ID替换为连续整型ID）；(3) 在线通过access/state/observation三类声明式原语执行交互。

  全栈执行例子（同PowerPoint任务 + DMI + GPT-5 medium + Windows）：
  - 算法/Agent层：LLM从prompt中core topology层级文本理解功能树结构→声明访问目标控件ID（`Blue`对应leaf ID 128、`Apply to All`对应leaf ID 132，shared subtree中的需附带entry_ref_id）→声明内容只含语义层面"我要访问什么"，不含"如何到达"。
  - 系统框架/Serving层：DMI executor接收声明式JSON→解析控件ID为XPath-like primary_id|control_type|ancestor_path→求解唯一导航路径（确定性图搜索，非LLM规划）→fuzzy matching当前窗口控件→按OK>Close>Cancel优先级关闭浮窗→沿路径点击导航控件→到达后通过UIA control patterns执行状态设置（SelectionPattern选色、InvokePattern点击Apply to All）。成功74.1%（+29.6%），步数4.61（-43.5%），时间239s（-39%）。失败81.0%为policy-level。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。

  **缺陷1：LLM必须生成完整GUI操作序列，视觉定位和精确交互是LLM弱项**
  → DMI将控件定位从LLM的视觉/文本推理转为确定性UIA树匹配（XPath-like: primary_id|control_type|ancestor_path）。不用LLM识别"蓝色按钮在哪里"，而通过UIA automation_id和ancestor_path做确定性映射。消融实验验证：为baseline注入DMI static navigation knowledge（文本描述或JSON），只要不启用声明式接口，SR反而从44.4%降到42.0%，证明**接口形式本身**决定性能——知识有用但不能通过命令式接口有效利用。

  **缺陷2：GUI-only对所有控件类型一视同仁，未利用UIA intrinsic semantics**
  → DMI对UIA控件分类处理：navigation控件（菜单、tab、dropdown）用DFS路径求解而非LLM规划；interaction控件通过control patterns做结构化状态读写（ScrollPattern替代拖拽坐标、TextPattern替代逐行选择、SelectionPattern替代多选点击）；transient窗口用OK>Close>Cancel优先级关闭以恢复导航状态。

  **缺陷3：长操作链级联失败，每步LLM推理延迟累计**
  → DMI通过state原语将多步复合交互压缩为单次声明：`set_scrollbar_pos(80%)`替代拖拽坐标定位、`select_lines(start, end)`替代多次点击+拖拽、`select_controls([id1,...])`一次性多选。interface lifting使步骤减少43.5%，降低累计延迟和级联风险。

  **缺陷4：全量5K+控件信息撑爆prompt context window**
  → DMI分层core topology压缩：默认只提供主干文本（Excel约30K tokens），通过`further_query`按需扩展缺失分支。连续整数ID替换冗长XPath-like ID进一步压缩token。

  **关键trade-off**：DMI以离线建模成本（每Office应用自动<3h + 人工约1.5人日）和在线prompt token开销（core topology 15K-30K tokens）换取执行稳定性和成功率。覆盖范围限于UIA可完整枚举的标准控件应用；自由绘图、精确位置调整、游戏/专业图形软件保留GUI slow-path fallback。需应用版本固定、UIA信息可用。
