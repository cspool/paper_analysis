论文标题：From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents


将Agent控制现有软件的面向人的接口（视觉定位，鼠标追踪等操作序列）整形为状态空间的建模和变换。

开源仓库确认：
    - 状态：已找到
    - 链接：https://github.com/dmi-interface/DMI
    - 说明：论文 Implementation 脚注明确写明 DMI 将发布在该 GitHub 地址；当前该仓库可访问，README 标题为 “DMI: LLM-friendly OS interfaces for Computer-Use Agents”，说明其集成到 Microsoft UFO Windows computer-use agent framework，并提供面向 Microsoft Office Word、Excel、PowerPoint 的预构建 core topology。仓库显示 MIT License，Python 代码，且 README 说明该仓库重点开源和复现 DMI execution layer。

1、论文工作：
    - 论文要解决的核心问题：LLM 驱动的 Computer-Use Agent 主要通过 API 或 GUI 与应用交互。API 路线成功率高、步骤少，但很多真实应用没有公开 API，通用性差；GUI 路线通用，但 GUI 是面向人的命令式接口，会迫使 LLM 把高层任务拆成大量细粒度动作，例如点击菜单、切换 tab、展开 dropdown、拖动滚动条、观察反馈再继续。这会放大 LLM 的弱项：视觉定位不稳定、坐标/拖拽精度差、每轮推理延迟高、长动作链容易级联失败。
    - 论文的主要贡献：论文提出 Declarative Model Interface（DMI），把已有 GUI 抽象成三类声明式原语：access、state、observation。核心思想是 policy-mechanism separation：LLM 只负责“语义上要做什么”的 policy，DMI 负责“如何导航到控件并完成交互”的 mechanism。DMI 不要求修改应用源码，也不依赖应用私有 API，而是基于 OS accessibility / Windows UI Automation（UIA）从 GUI 中抽取控件、导航关系和可调用 control pattern。
    - 论文所处背景：该工作属于 OS interface、LLM agent、computer-use agent 和 GUI automation 交叉方向。论文把 LLM 视为一种新的 OS 用户：它有大上下文和结构化输出能力，但视觉与高频闭环控制弱，因此不适合被迫模仿人类使用 GUI。目标场景是 Windows 上的复杂 Office 应用，包括 Word、PowerPoint、Excel，这些应用控件数超过 5K，存在嵌套对话框、动态面板、深层菜单、循环导航和 merge node。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：论文采用 Microsoft UFO-2 的 GUI-only 模式作为主要 baseline，并使用其 action sequence 能力减少 round trip。Baseline 的根本问题不是没有 UIA 信息，而是接口仍然是命令式的：LLM 必须输出可见控件上的点击、键盘、坐标拖拽等动作序列。即使给 baseline 额外加入 DMI 的 navigation forest 知识，只要不启用声明式接口，GPT-5 medium 的成功率仍从 44.4% 降到 42.0%，步骤也没有减少，说明静态知识本身不是主要收益来源。机制层面，baseline 失败大量来自控件定位、导航和复合交互，例如找不到正确控件、路径选择错误、拖拽/选择失败。
    - 论文的设计方法：DMI 先离线构建 UI Navigation Graph（UNG），节点是 UIA 暴露的控件，边表示点击一个控件后可达的新控件；再把有环、有 merge node 的图转成 path-unambiguous forest，包含 main tree 和 shared subtrees。在线时，DMI 给 LLM 一个压缩的层级文本描述和 query-on-demand 机制。LLM 不再输出完整导航路径，而是声明目标控件 ID、目标状态或观测请求；DMI 通过 `visit`、`set_scrollbar_pos`、`select_lines`、`select_controls`、`get_texts` 等接口完成执行。
    - 方法如何对冲 Baseline 缺陷：对 navigation，DMI 将“找到并打开目标控件”从 LLM 的长序列规划变成确定性图路径求解；对 interaction，DMI 利用 UIA control patterns 把滚动、文本选择、多选、文本读取等复合操作包装成状态设置或结构化观察；对 context，DMI 不把全部 5K+ 控件一次性塞入 prompt，而是默认提供 core topology，并通过 `further_query` 扩展缺失分支；对不稳定 UI，DMI 使用 fuzzy matching、结构化错误反馈、加载重试和非叶节点过滤，减少由于控件名称变化、LLM 输出导航节点、控件暂未出现造成的失败。
    - 关键 trade-off：DMI 用离线建模和部分人工配置换取在线执行稳定性。构建 UNG 需要应用版本固定、UIA 信息可用，并且要维护 access blocklist 和 context-aware exploration 的代表对象；论文报告每个 Office 应用自动建模小于 3 小时，人工约 1.5 人日。DMI 还增加 prompt token，core topology 大约为 Excel 30K、Word 15K、PowerPoint 15K tokens，但通过减少 LLM 调用轮次降低总 token。DMI 不能覆盖自由绘图、精确位置调整、游戏/专业图形软件等非标准控件操作，因此保留 GUI imperative path 作为 slow-path fallback。

3、论文实现：
    - Baseline 如何实现：Baseline 是 UFO-2 的 `UFO2-base + action sequence`，论文记为 UFO2-as。它是 GUI-only，不使用 Office-specific COM API，因为 COM API 不具备通用性。为了公平，baseline 和 DMI 设置都注册 UIA event handler，使应用暴露完整 control tree，避免 lazy loading 造成差异。Baseline 在调用 LLM 前给 accessible-tree 控件打 alphabetic labels（如 A、HF）并写入 prompt；action sequence 允许一次输出多个动作，但这些动作仍受限于当前可见 UI 控件。
    - 新设计如何实现：DMI 原型超过 18K 行 Python，基于 pywinauto 调用 Windows UI Automation。UNG 构建使用 DFS 和 differential capture：先采集当前 accessibility tree，点击候选控件后再次采集，新出现的控件构成导航边；新顶层窗口或 modal window 通过 process_id 和 window listener 检测。控件 ID 采用 XPath-like 形式：`primary_id|control_type|ancestor_path`，其中 primary_id 优先用 UIA automation_id，不存在则退化到控件名称或 `[Unnamed]`。生成 forest 后，DMI 用 `name(type)(description)_id[children]` 的紧凑结构序列化拓扑，并把冗长控件 ID 替换成连续整数 ID。
    - 实验 / 实现平台：实验对象是 Microsoft Word、Excel、PowerPoint，使用 Microsoft 365 builds；任务来自 OSWorld-W（Windows）中 27 个单应用 Office 场景，排除多应用任务，因为这需要额外建模 OS 控件。LLM 使用 OpenAI GPT-5 与 GPT-5-mini，reasoning effort 包括 minimal、medium 等；主要对比 GPT-5 medium、GPT-5 minimal 和 GPT-5-mini medium。每个任务最多 30 步，运行 3 次取平均，不做模型微调。
    - 关键实验设置与指标：指标包括成功率 SR、平均 LLM 调用步数 Steps、完成时间 Time，并只在成功 case 上计算 steps/time。核心结果中，GPT-5 medium 下 GUI-only SR 为 44.4%，GUI+DMI 为 74.1%，绝对提升 29.6%，相对提升 1.67x；平均步数从 8.16 降到 4.61，减少 43.5%；完成时间从 392s 降到 239s，减少约 39%。GPT-5 minimal 下 SR 从 23.5% 到 40.7%，steps 从 8.42 到 5.52；GPT-5-mini medium 下 SR 从 17.3% 到 43.2%，约 2.5x。失败分析中，GUI+DMI 的失败 81.0% 是 policy-level，而 GUI-only 的 mechanism-level 失败占 53.3%，说明 DMI 确实把主要错误从导航/交互机制转移到语义理解和规划。

4、pipeline/kernel解析：
    - 新pipeline/kernel是什么：论文没有提出 GPU kernel、算子融合或硬件执行单元；它提出的是 DMI 的 declarative GUI execution pipeline。这个 pipeline 分为离线建模和在线执行两段：离线段从应用 UIA tree 中抽取 UNG，消环并处理 merge node，生成 path-unambiguous forest，再压缩成 LLM 友好的 core topology；在线段让 LLM 通过 access/state/observation 原语声明意图，由 DMI executor 解析目标 ID、求解唯一导航路径、匹配当前 UI 控件、执行点击/输入/状态设置/文本观测，并在必要时回退到原始 GUI slow path。
    - 新pipeline/kernel的执行流例子：以 PowerPoint “把所有幻灯片背景设为蓝色”为例，GUI-only baseline 需要 LLM 规划并执行类似点击 Design、点击 Format Background、选择 Solid fill、打开 Fill Color、选择 Blue、点击 Apply to All 的可见控件序列；每个中间步骤都可能因控件不可见、识别错误或状态变化失败。DMI 中，离线 forest 已经保存这些控件之间的导航关系和路径歧义处理结果。在线时，LLM 只需要在 prompt 中根据层级描述选择目标功能控件，例如声明访问 `Blue` 和 `Apply to All` 对应的 leaf ID；如果目标在 shared subtree 中，则同时给出 `entry_ref_id`。`visit` 接口收到 JSON 数组后先过滤非叶导航节点和不一致 shortcut，解析 root-to-target path，从当前窗口开始反向匹配可见层级；若不在当前窗口则按 OK > Close > Cancel 的优先级关闭浮窗以回到可导航状态；随后沿路径点击对应控件，必要时输入文本或执行快捷键。对于滚动条或文本选择这类复合交互，LLM 不输出拖拽坐标，而调用 `set_scrollbar_pos(80%)` 或 `select_lines(start, end)`，DMI 通过 UIA ScrollPattern / TextPattern / SelectionPattern 设置目标状态并返回结构化状态。对于 Excel 单元格或表格内容，`get_texts()` 在 passive mode 会在每次 LLM 调用前采集 DataItem 的截断结构化文本，active mode 则可按需返回完整文本，从而减少像素级 OCR 和额外 observe-act 循环。
