idea-answer-skill.

对每个idea.md
1. 一级上下文: 用idea的关联论文标题, 搜索exp notes, knowledge notes下的md作为上下文.
2. 二级上下文: paper-secs的论文标题path中的md作为上下文, 关键词omini搜索.
3. 三级上下文: 对于不明确的概念或实现/方法, 用关键词搜索(obsidian-keyword-explain).
4. 优先一级, 推理前进, (新的输出)过程中信息不足时二级和三级, 推理前进, (新的输出)过程中信息不足时二级和三级, ...
5. 回答问题(idea-question-skill的蒸馏, 提出问题)
6. 输出格式


idea-question-skill.(人工归纳, mark ts)
按照review_draft.md, 和L*_horizon_summary.md的差异学习输出内容(问题).
1. review_draft中哪些##类别, 这是对我有价值的信息(大问题).
2. 每个类别下, 有很多方法类别, 这些方法类别是值得提取/注意的(大问题的答案).
3. 每个方法的描述, 关注我用('****')加粗强调的部分, 这是我关注的方法核心, 是该方法值得注意后需要深入挖掘获取的信息(进阶问题).
4. 先理解方法, 再针对review, 学习正例标准.


question-data-mark
将summary的输出, 人工归纳, 先写review, 细化.
低优先级/抛弃的标准: 用不包含在review的方法项目, 学习反例.
human review和补充



idea-answer-summary-gap
架构故事思路:
架构设计侧重模块化, 什么计算需要什么模块, 基础模块提供通用性能保底, 专用模块补齐特化性能, 专用模块提供插件效果.
自动review(筛选, 价值), 深入探索(性能收益,gap,空间)