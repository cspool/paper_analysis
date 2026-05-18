## Windows UI Automation (UIA)

术语是什么？

Windows UI Automation (UIA) 是 Microsoft Windows 操作系统的 accessibility framework，为辅助技术和自动化工具提供对UI控件的编程访问。UIA通过provider-client架构暴露每个GUI控件的结构化信息：control type（Button、Edit、ListItem、Tab等）、automation ID、名称、层级父子关系、支持的control patterns（InvokePattern用于点击、ScrollPattern用于滚动、TextPattern用于文本操作、SelectionPattern用于选择、ValuePattern用于读写值等）。UIA还提供event机制（structure changed、window opened/closed、focus changed等），供client订阅UI状态变化。与纯截图/OCR方式不同，UIA提供程序化的控件访问，不需要视觉定位。

从系统架构角度拆解术语：

UIA在DMI系统中的角色：

1. **离线UNG构建**：DMI通过UIA的`IUIAutomationTreeWalker`遍历应用accessibility tree，采集每个控件的control type、automation_id、name、supported patterns和层级关系。Differential capture通过UIA event handler检测新窗口/控件出现。

2. **在线控件匹配**：DMI executor通过UIA从当前桌面查找目标控件——使用XPath-like路径 `primary_id|control_type|ancestor_path`，其中primary_id优先使用UIA的automation_id（最稳定），不存在时退化为控件名称。fuzzy matching处理控件名称的微小变化。

3. **交互执行**：DMI不通过模拟鼠标/键盘操作，而是调用UIA control patterns：(a) `InvokePattern.Invoke()`替代点击Button；(b) `ScrollPattern.SetScrollPercent()`替代拖拽滚动条；(c) `TextPattern.GetSelection()`/`SelectionPattern.GetCurrentSelection()`读取选择状态；(d) `ValuePattern.SetValue()`设置Edit控件的文本。

4. **UIA限制**：非标准控件（自绘UI、游戏、专业图形软件）的UIA信息不完整或不准确，此时DMI回退到GUI imperative path。

术语一般如何实现？如何使用？

UIA的主要API（C++/COM）：`IUIAutomation::ElementFromHandle()`从窗口句柄获取UIA element；`IUIAutomationElement::FindAll(TreeScope, Condition)`按条件和范围搜索子控件；`IUIAutomationElement::GetCurrentPatternAs()`获取control pattern接口。Python绑定通过`pywinauto`（基于`comtypes`/`uiautomation`）或`uiautomation`库使用。微软的Power Automate和Windows Narrator均基于UIA实现。在Computer-Use Agent中，UIA比screenshot+OCR提供更可靠的控件识别和状态读取。论文重要实现细节：baseline和DMI设置都注册UIA event handler，使应用暴露完整control tree（避免lazy loading造成差异），保证公平对比。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---
