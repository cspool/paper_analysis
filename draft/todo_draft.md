1. 整理 /data3/agent_research中的论文自动下载， /home/descfly/Desktop/marker中的md转换脚本， 不需要迁移环境， 只需要分别封装一个接口脚本（自动下载：提供文件内标题批下载， 直接提供标题的下载。 md转换， 批量转换， 指定单个论文的转换）， 能在当前目录使用他们的功能。
2. 在当前目录整理所有脚本（‘/data3/paper_analysis/scripts’）的使用说明（补充到readme）， run_all_papers， paper_mdspilt， repo_split， 通过temp目录来进行单个论文的临时处理进行说明和测试。



3. .claude/下的Claude Code中伪代码风格的a和q skill和调度脚本的说明, 总结成blog风格（开头的效果展示， 然后思路介绍（包含skill的work flow概述）， 然后伪代码风格skill的实例）。

4. 总结容器启动的参数需求/注意：1. 只读共享的依赖目录，避免重复安装和下载，防止容器中删除host的重要包。 2. host安装工具，提供容器使用，需要bind配置或数据（claude，codex， skill等）。 3. host和容器共享的volume（项目，nsys，ncu等工具）， 直接bind， 但注意容器登陆身份，避免volume被容器锁住（只允许root/容器中使用）。 4. 驱动软件需要host下放用户权限（允许非root使用），previledge的权限过高（很危险）， 赋予特定文件的读取或函数调用权限。

5. 长线任务中，将配置记入/memory， memory需要放什么？编程的Spec。
   1. 本地model cache忽略， 不认识。
   2. GPU 1 prefer遗忘。
   3. 不要llm从头编写， 而是让其学习eval（经过验证的推理过程代码）的调用链。