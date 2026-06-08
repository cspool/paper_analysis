不需要逐个论文, learning等价于头脑风暴/关联性survey来缩小范围, review-answer-question则是对缩小范围后的paper逐个精读.

idea-answer-skill.

对指定<idea_note_path>的idea.md
1. 一级上下文: 用idea的关联论文标题作为关键词, 搜索'/data3/paper_analysis/experiment_notes', '/data3/paper_analysis/knowledge_notes'下的<title>.md作为上下文(论文限定).
2. 二级上下文: '/data3/paper_analysis/paper_secs/'下搜索(粗略匹配)论文标题命名的路径<title_path>, <title_path>下按关键词用omini搜索关联的md文件作为上下文.
3. 三级上下文: 按照obsidian-keyword-explain的skill要求, 用关键词搜索上下文.
4. 优先一级, 推理前进, (新的输出)过程中信息不足时二级和三级, 推理前进, (新的输出)过程中信息不足时二级和三级, ...
5. 回答问题(idea-question-skill的蒸馏, 提出问题)
6. 输出格式


idea-question-skill.(人工归纳, mark ts)

question agent和answer agent问答来判别, question(当前skill)提问, 接收回答, 直到作出判别.按照review_draft.md(人工筛选过程/正样本)学习如何判别和提问. 
1. review_draft中哪些##类别, 这是对我有价值的信息(大问题).
2. 每个类别下, 有很多方法类别, 这些方法类别是值得提取/注意的(大问题的答案).
3. 每个方法的描述, 关注我用('****')加粗强调的部分, 这是我评估方法所用的关注点, 是该方法值得注意后需要深入挖掘获取的信息(进阶问题).
4. 先理解方法, 再针对review, 学习正例标准(高相关性/深入价值).
5. review_draft.md是对L[1 |2 | 3 | 4 | 5 | 6]_horizon_summary.md的人工筛选, 被去除的是反例(不包含因为重复被删除的正例).
6. question agent完成后增加输出要求, 将评判高价值/相关性的方法或实现生成摘要条目(方法/实现, review, 来源)记录在<review_summary>.md中(重复或近似的条目, 将原有条目升级类别后, 作为子条目)


idea-review ts
检查2个role skill和脚本, 对齐我设想的调度: 
0. 脚本维护2个agent的运行状态(当前输出完成, 等待后续输入信号)来进行调度, 2个agent的输入和输出信息(时间戳+agent+输入/输出+内容), 都需要形成文档, 下面不在明确提记录的要求. 
1. 脚本启动2个role session并让其加载技能(下称q agent和a agent, 将review skill改名字为idea-answer): 
   1.1 注意启动时将skill内容读取作为prompt, 并明确等待后续输入, 这样能避免递归启动. 
   1.2 我注意到脚本开启session前有额外的prompt描述, 请把他们更新到skill, 脚本只负责调度agent和转发agent输出. 
2. 脚本等待2个agent给出等待输入的信号后(初始化完成), 给q agent增加输入'开始提问'并等待输出完成. 
3. 脚本接受q agent输出完成信号, 将q agent输出转发给a agent作为增加输入, 等待a agent输出完成. 
4. 脚本接受a agent输出完成信号, 将a agent的输出转发给q agent作为增加输入, 等待q agent输出完成. 
5. 循环3-4, 直到q agent输出完成中包含评判完成的信息, 并给出最终review和保存review, 结束2个session. 
6. q agent负责梳理和对齐我所关注的内容(skill中已经包含)来作出评估决策和生成review, a agent负责被引导回答和包含论文和知识的上下文.



idea-gap
架构故事思路:
架构设计侧重模块化, 什么计算需要什么模块, 基础模块提供通用性能保底, 专用模块补齐特化性能, 专用模块提供插件效果.
自动review(筛选, 价值), 深入探索(性能收益,gap,空间)

现有的结果需要深入.在深入过程进行总结标准, review时选择/评判标准(理由, 专一), deep是深入学习方向/或问题( 结合reviw, 自由获取相关上下文, 分析产生收益的原因和细节?), gap是扩展和深入研究的内容(结合全局)


多agent的状态管理, 能替换slm?slm如何知道llm的状态.