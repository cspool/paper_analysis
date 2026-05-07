根据提供的论文标题或论文编号，在当前工作目录搜索对应论文（一篇论文），将其作为上下文进行阅读和分析，回答下列问题。

## 核心提取维度

### 1. 核心贡献（Core Contribution vs. Baseline）

#### 1.1 痛点识别（Problem Addressed）

基于论文原文中的 Abstract、Introduction / Background / Motivation 和 Method，总结 Baseline 存在的具体技术缺陷。需要回答：

- 原有方法、系统、硬件架构或其他的瓶颈是什么？
- 该瓶颈来自计算、访存、通信、调度、数据布局、稀疏性、精度、并行性、能耗、面积，还是其他因素？
- 论文认为已有方法为什么不够好？
- 该问题在什么应用场景、模型规模、硬件平台或系统条件下变得突出？

#### 1.2 设计对冲（Methodological Design）

基于论文原文中的 Motivation、Background 和 Method，说明论文通过什么具体设计解决上述痛点。需要回答：

- 方法的核心机制是什么？
- 它如何对应解决 Baseline 的缺陷？
- 是否引入新的数据流、硬件单元、调度策略、编译优化、稀疏模式、近似计算、缓存策略、训练 / 推理策略或系统协同机制？
- 设计中有哪些关键 trade-off？

### 2. 论文实现（Implementation）

基于论文 Implementation / Experiment / Evaluation 部分总结：

- 论文如何实现 baseline；
- 论文如何实现所提出的创新设计；
- 使用了哪些硬件平台、模拟器、框架、模型、数据集或 benchmark；
- 实验中的关键参数、实现假设和限制是什么；
- 性能、能耗、面积、吞吐、延迟、精度、成本或可扩展性收益如何被测量；

### 3. 提取和解析论文设计的新pipeline或kernel

基于论文原文中的 Motivation、Method和Implementation，提取和聚焦解析论文提出的新pipeline或kernel打包方式。
- 新pipeline/kernel是什么，简单易懂表达；
- 新pipeline/kernel的执行流是什么，尽量用例子表达；

---

## 输出格式

以论文名字新建并输出到 **一个 Markdown 文件**。

文章按照“字典式结构”输出，格式如下：

```md
论文标题：<Paper Title>

    开源仓库确认：
        - 状态：已找到 / 疑似相关 / 未找到明确开源仓库 / 无法确认
        - 链接：<URL 或 N/A>
        - 说明：<官方仓库、非官方复现、作者主页、artifact 页面等判断依据>

    1、论文工作：
        - 论文要解决的核心问题：<内容>
        - 论文的主要贡献：<内容>
        - 论文所处背景：<内容>

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：<内容>
        - 论文的设计方法：<内容>
        - 方法如何对冲 Baseline 缺陷：<内容>
        - 关键 trade-off：<内容>

    3、论文实现：
        - Baseline 如何实现：<内容>
        - 新设计如何实现：<内容>
        - 实验 / 实现平台：<内容>
        - 关键实验设置与指标：<内容>

    4、pipeline/kernel解析：
        - 新pipeline/kernel是什么：<内容>
        - 新pipeline/kernel的执行流例子：<内容>
```

---