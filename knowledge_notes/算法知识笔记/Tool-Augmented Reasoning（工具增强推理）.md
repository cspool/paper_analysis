## Tool-Augmented Reasoning（工具增强推理）

术语是什么？通过联网搜索让回答具体和精准。

Tool-Augmented Reasoning是AI Agent区别于静态LLM推理的核心能力。在tool-augmented reasoning中，agent不仅进行内部语言推理（reasoning），还调用外部工具（tool use）获取实时数据、执行非语言操作（计算、代码执行、web搜索）。工具调用结果以observation形式返回，被纳入agent上下文中指导后续推理。这种reasoning-acting-observing循环使agent能处理超出LLM训练数据覆盖范围的任务。论文量化了这一范式的系统成本：LLM inference占总延迟69.4%、tool execution占30.2%，两者因sequential dependency难以overlap。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Tool-augmented reasoning的依赖链
LLM_output = f(context)           // 输出包含tool selection和parameters
Tool_result = execute(tool, params)  // 必须等LLM_output确定tool和params
Next_LLM_input = concat(context, LLM_output, Tool_result)  // 必须等Tool_result
// → LLM inference和tool execution难以并行（数据依赖）
```

论文量化了系统成本：(1) Tool latency因类型差异大：Wikipedia API ~1.2s/call, WebShop本地web ~20ms/call, Wolfram Alpha API中等；(2) GPU在tool执行期间最多54.5% idle time（HotpotQA/MATH的CPU/外部tools）；(3) Tool history tokens是context膨胀的主要来源——知识密集型任务中tool返回大段文本；(4) LLMCompiler通过DAG planning尝试overlap planning和tool execution，但仅实现18.2% overlap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Tool system实现：(1) Tool定义：每个tool有name、description、parameters schema（供LLM理解）；(2) Tool Executor：解析LLM output中的tool call→验证参数→执行（本地Python函数或外部API）→返回结构化result；(3) Tool可以用OpenAI function calling API、Anthropic MCP协议或Google A2A协议标准化；(4) 本地tools（code interpreter, calculator）延迟低不占外部带宽，远程tools（Wikipedia API, Wolfram Alpha）延迟高有网络依赖。论文在AgentBench中集成Wikipedia search/lookup APIs、WebShop navigation、Wolfram Alpha API、Python executor等tools。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

---

