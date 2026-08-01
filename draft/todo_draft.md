1. 整理 /data3/agent_research中的论文自动下载， /home/descfly/Desktop/marker中的md转换脚本， 不需要迁移环境， 只需要分别封装一个接口脚本（自动下载：提供文件内标题批下载， 直接提供标题的下载。 md转换， 批量转换， 指定单个论文的转换）， 能在当前目录使用他们的功能。
2. 在当前目录整理所有脚本（‘/data3/paper_analysis/scripts’）的使用说明（补充到readme）， run_all_papers， paper_mdspilt， repo_split， 通过temp目录来进行单个论文的临时处理进行说明和测试。



3. .claude/下的Claude Code中伪代码风格的a和q skill和调度脚本的说明, 总结成blog风格（开头的效果展示， 然后思路介绍（包含skill的work flow概述）， 然后伪代码风格skill的实例）。

4. 总结容器启动的参数需求/注意：1. 只读共享的依赖目录，避免重复安装和下载，防止容器中删除host的重要包。 2. host安装工具，提供容器使用，需要bind配置或数据（claude，codex， skill等）。 3. host和容器共享的volume（项目，nsys，ncu等工具）， 直接bind， 但注意容器登陆身份，避免volume被容器锁住（只允许root/容器中使用）。 4. 驱动软件需要host下放用户权限（允许非root使用），previledge的权限过高（很危险）， 赋予特定文件的读取或函数调用权限。

5. 长线任务中，将配置记入/memory， memory需要放什么？编程的Spec。
   1. 本地model cache忽略， 不认识。
   2. GPU 1 prefer遗忘。
   3. 不要llm从头编写， 而是让其学习eval（经过验证的推理过程代码）的调用链。



重新梳理我的需求: 

1. 知识库是从HPC领域的最新科研论文中提取的多维度场景,知识,方法,实现的仓库, 每个仓库有自己的表达维度(experiment, idea&baseline, knowledge， human). 
   1.1 不同维度笔记分别作为不同需求的查询目标， 查询使用omini-search的tool， 来自obsidian。
   1.2 用什么关键词查询？什么时候触发查询？

2. 我会给你一个topic, 从知识库中获取topic相关的进一步优化性能的潜力， 这是最终产出。
   2.1 过程输出按照实际需求， 但最终输出需要提供人阅读理解。
   2.2 过程输出定义， 分为agent和脚本交互的协议，格式化数据给agent输入思考。


3. 6L是可能修改来带来性能优化的方式, topic则限定其中的子集,工作流是AI自动挖掘潜力的过程, 本质是在topic范围内搜索. 
   3.1 archor是为了圈定topic设置的元数据，看作L6中的一个区域中心， 来延伸子空间， archor是从加速场景-baseline这个核心角度来定义。archor如何定义，如何表达？
   3.2 direction是archor定义L6子空间中可能的方向， 需要探索和审阅可能的优化方向。direction如何定义，如何表达？
   3.3 workflow最好使用loop的形式， 因为依次搜索-思考-审阅，难以得到足够多的信息。不需要有优先设计，完成其他定义后进行脚本-agent编排拆分即可。 

一般而言，调度脚本的作用有二：
1. 解决持久化agent在上下文增长时， 丢失运行状态， 因此让脚本充当持久记忆但没有智能的调度器（agent编排器）， 设置决策agent让脚本根据条件触发来得到决策。
2. goal风格agent或单次agent的顺序调度， agent完成goal后， 脚本启动下一个agent， agent的skill自行维护好需要存储的输出。


agent映射白名单：
19，20,21合并： 在worker，reviewer，脚本中体现， 但是查询缺口由loop的reviewer负责。
23：保持目前。
24：在worker，decision和脚本中体现。
25：保留。
28，29合并：保留。但注意：topic的6L空间应该是archor集合来定义，并且随着loop动态增长。
30：保留。
31：保留。
33-35保留。

按白名单修改闭合需求方式， 并从loop视角展示如何闭合需求。


1. 调度面：Script ↔ Decision
     最终需求，workflow描述（完成需求的方法和策略），自身职能的执行过程，目标，约束写入decision skill， 并在调用时将'使用${skill_name}'加入prompt。
     调用时的输入形式是字符串prompt，是任务当前状态('任务当前状态'+{所在路径})、2类agent的已校验结论('2类agent的当前结论'+{所在路径})、本次允许决策('允许决策'+{字面量})；
     
     输出建议使用协议字段包装， 而不是json， 来降低agent输出精度的要求， 输出是：
     decision = WORKER | REVIEWER | COMPLETE
     guidance = 可选， 包含对决策启动的补充解释和需求prompt（精简）
     且decision需要脚本从输出关键词匹配， 增加鲁棒性。

  2. 内容面：Script ↔ Worker / Reviewer
      archor，direction定义和用途，自身职能的执行过程，目标，约束写入worker skill， 并在调用时将'使用${skill_name}'加入prompt。
      审阅的方法和标准，自身职能的执行过程，目标，约束写入reviewer skill， 并在调用时将'使用${skill_name}'加入prompt。
     Worker和Reviewer调用时的输入是prompt（同和decision的交互），包含当前2类agent的结论、任务覆盖要求（聚焦archor+direction）、decision的附加guidance。
     Skill ， Ref和tool 按 Agent 类型独立装配， skill中可以引用ref增强专业性（专家知识，可能包含6L划分/例子），推荐引用tool减少任务无关错误（根据输入关键词自动包装json格式的工具脚本， 调用ominisearch的工具脚本 等等）。

   3.持久化面：Script 内部
     按照约定好的协议和json格式，以时间线或轮次，保存完整运行状态、agent原始输出、校验结果和事件；这些不要求 Agent 回显。
     

 ### 1. Decision 调度面

  Decision Skill 保存稳定内容：

  - Workflow 的目标、完成方法和整体策略；
  - Decision 的职能和判断过程；
  - WORKER / REVIEWER / COMPLETE 的含义；
  - guidance 的边界；
  - 输出协议和自检方法。

  本次 topic、运行状态和动态验收进度不应写入 Skill，而应存在 Script 生成的状态快照中，否则更换 topic 就要修改 Skill。

  调用 Prompt 可以保持为：

  使用 $decision_skill_name

  任务当前状态：<immutable_state_snapshot_path>
  两类 Agent 的当前已校验结论：<validated_conclusions_path>

  [ALLOWED_DECISIONS]
  WORKER
  REVIEWER

  [OUTPUT_PROTOCOL]
  decision = <一个允许的关键词>
  guidance = <可选精简说明>

  推荐使用简单行协议：

  decision = WORKER
  guidance = 增加一个 Direction，重点补足延迟与吞吐约束。

  Script 不应在整段文本中无约束搜索关键词，否则 guidance 可能同时出现 WORKER、REVIEWER 等词。推荐解析策略是：

  1. 优先匹配 decision = ... 或 decision: ... 行；
  2. 忽略大小写、空格和 Markdown 列表符；
  3. 只接受本次允许集合中的独立关键词；
  4. 必须恰好得到一个唯一值；
  5. 找不到、得到多个或不在允许集合中时，携带错误说明重试；
  6. guidance 不参与 Decision 解析。

  这样既不要求严格 JSON，又不会因全文关键词碰撞而误调度。

  ### 2. Worker / Reviewer 内容面

  Skill 分工同意你的设计：

  - Worker Skill：Anchor、Direction 的定义和用途，内容工作流程、目标、约束；
  - Reviewer Skill：审阅方法、标准、流程、目标和约束；
  - 专业 Ref：6L、领域知识、示例、评审标准；
  - 工具：结果包装、检索、来源读取等执行辅助。

  调用 Prompt 最小包含：

  使用 $worker_or_reviewer_skill_name

  当前两类 Agent 的已校验结论：<current_conclusions_path>
  本 Turn 任务覆盖要求：<turn_coverage_path>
  Decision 附加说明：<optional_guidance>

  其中“任务覆盖要求”必须明确绑定当前 Anchor 或 Direction，避免 Agent 自行改变工作对象。

  工具包装方面，建议根据 Script 已确定的 job kind 选择 JSON 模板，例如 Anchor、Direction、Review，而不要让工具根据自然语言
  关键词猜测结果类型。关键词可以辅助填充内容，但结果种类必须由 Script 确定。

  ### 3. Script 持久化面

  推荐同时保存三类数据：

  canonical state snapshot
  append-only events timeline
  per-round / per-attempt records

  每个 Turn 至少保存：

  - 实际 Prompt；
  - Skill 名称；
  - 输入快照路径；
  - Agent 原始输出；
  - Script 解析结果；
  - 校验结果；
  - 重试错误；
  - 最终提交的结论或 Artifact。

  状态和结论路径应指向本次调用冻结的不可变快照，不能指向执行期间会变化的通用 current.json。Agent 不需要回显 round、
  attempt、hash 等绑定字段，由 Script 在调用外层记录。

  Decision 的交互格式已经可以基本冻结。下一步最需要讨论的是 Worker 和 Reviewer 的输出方式：是直接返回内容 JSON，还是由工具
  写入 Script 预分配的结果文件，然后 Agent 只返回一个简短完成协议。