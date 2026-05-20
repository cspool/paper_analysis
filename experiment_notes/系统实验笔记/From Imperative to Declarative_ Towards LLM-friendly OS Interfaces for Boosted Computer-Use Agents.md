## From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现Declarative Model Interface (DMI)，一个LLM友好的OS声明式接口中间层，基于UFO-2 agent框架修改。核心实现分两段：(1) 离线阶段：通过Windows UI Automation (UIA) 从目标应用中抽取UI Navigation Graph (UNG)，带环和merge node的图转为path-unambiguous forest，再压缩为LLM友好的core topology层级文本描述（Excel约30K tokens, Word约15K, PowerPoint约15K）；(2) 在线阶段：DMI为LLM提供三类声明式原语——access（声明目标控件ID）、state（设置控件状态如滚动条位置、选中范围）、observation（读取文本），由DMI executor负责导航路径求解、控件模糊匹配和交互执行。LLM不再生成细粒度GUI操作序列（点击、拖拽、输入），只需声明意图。DMI原型超18K行Python，基于pywinauto调用Windows UI Automation。
  实验比较：(a) UFO-2 GUI-only baseline（UFO2-as, 使用action sequence）vs GUI+DMI，在OSWorld-W中27个单应用Office场景下对比成功率(SR)、平均LLM调用步数(Steps)、完成时间(Time)；(b) 不同LLM模型下的DMI增益：GPT-5 medium/minimal, GPT-5-mini medium；(c) 消融实验：为baseline注入DMI navigation forest静态知识但不启用声明式接口，SR反而从44.4%下降到42.0%；(d) 失败分析：将失败分为policy-level（语义规划错误）和mechanism-level（控件定位/导航/交互错误），验证DMI是否将错误从mechanism转移到policy。

- 硬件平台是什么，配置是什么。
  实验平台为Windows系统，运行Microsoft 365 builds的Word、Excel、PowerPoint。LLM使用OpenAI GPT-5和GPT-5-mini API（reasoning effort: minimal/medium）。论文未明确说明具体Windows机器硬件配置（CPU/RAM等）。每个任务最多30步，运行3次取平均，不做模型微调。

- 开源Serving框架是什么。修改了什么。
  基于Microsoft UFO-2 open-source computer-use agent framework（https://github.com/microsoft/UFO）。修改内容：(1) 添加DMI execution layer，在UFO-2 agent workflow的原GUI action path之外增加declarative action path；(2) DMI层内实现UNG构建模块（DFS + differential capture发现导航边、merge node处理、shared subtree识别）、forest压缩模块（core topology生成、控件ID整型映射）、在线executor模块（path resolution、fuzzy matching、load重试、OK>Close>Cancel优先级的浮窗关闭策略、fallback到GUI slow-path）；(3) baseline和DMI都注册UIA event handler确保完整control tree暴露，避免lazy loading造成差异。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/dmi-interface/DMI（MIT License）。仓库集成到Microsoft UFO Windows computer-use agent framework，提供面向Microsoft Office Word、Excel、PowerPoint的预构建core topology。
  DMI作为Agent-OS交互中间层使用：
  1. 离线构建：每个应用运行UNG builder（自动exploration <3h，人工配置约1.5人日）→生成core topology→保存forest。
  2. 部署：在UFO-2中配置DMI executor，加载预构建core topology。
  3. 任务执行流（以PowerPoint "将所有幻灯片背景设为蓝色"为例）：
    a. Task到达UFO-2→LLM接收prompt（task + core topology层级描述 + 声明式API说明）。
    b. LLM生成声明式action JSON：声明控件ID（如`Blue`选项leaf ID、`Apply to All` leaf ID，shared subtree中需附带entry_ref_id）。
    c. DMI executor：解析控件ID→求解唯一导航路径（确定性图搜索）→fuzzy matching当前窗口控件→按OK>Close>Cancel优先级关闭浮窗→沿路径点击导航控件→通过UIA ScrollPattern/TextPattern/SelectionPattern等control patterns执行状态设置。
    d. DMI在每次LLM调用前通过passive mode采集DataItem截断文本，`get_texts()` active mode按需返回完整文本。
  4. GPT-5 medium下：SR 44.4%→74.1%（+29.6%绝对），Steps 8.16→4.61（-43.5%），Time 392s→239s（-39%）。GPT-5-mini medium：SR 17.3%→43.2%（约2.5×）。失败分析：GUI+DMI失败81.0%为policy-level，GUI-only mechanism-level失败占53.3%。
  5. DMI保留GUI imperative path作为slow-path fallback用于非标准控件场景。

