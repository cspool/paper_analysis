## ReAct (Synergizing Reasoning and Acting)

术语是什么？通过联网搜索让回答具体和精准。

ReAct（Reasoning + Acting）是一种AI agent workflow，将LLM的推理能力和行动能力交替结合。在每个step中，LLM生成一个thought（推理当前状态和下一步计划）和一个action（指定要调用的工具和参数）；系统执行action后返回observation（工具执行结果）；observation被追加到context后进入下一轮thought/action循环。ReAct agent持续这个循环直到生成final answer或达到iteration limit。ReAct是所有tool-augmented agent（Reflexion、LATS）的基础workflow。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ReAct单步pipeline（以HotpotQA Wikipedia search为例）：

```
Prompt = "You are an assistant solving QA tasks.
  You have access to Wikipedia search and lookup tools.
  Use Thought/Action/Observation format."

当前Context = [Prompt, UserQuery, History(LLM_outputs, Tool_results)]
llm_output = LLM(当前Context)
// llm_output格式: "Thought: <reasoning>\nAction: <tool>[<params>]\n"
// 或 "Thought: <reasoning>\nFinal Answer: <answer>"

action, params = parse(llm_output)
if action == "Final Answer":
    return params
else:
    tool_result = execute_tool(action, params)  // e.g. Wikipedia API (~1.2s)
    History.append(("Action: " + action, "Observation: " + tool_result))
    goto next_step  // input tokens增长（LLM history + Tool history）
```

论文测量：ReAct的LLM和tool call数量相近，tool latency因workload而异——WebShop本地web工具约20ms/call，HotpotQA Wikipedia API约1.2s/call。ReAct在HotpotQA的95th percentile latency为20.7s（vs ShareGPT 9.7s），WebShop 50.8s。prefix caching对ReAct的多轮LLM call特别有效（prefill latency降低60.1%）。ReAct在accuracy-latency trade-off中cost-efficiency表现均衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

ReAct通过prompt engineering实现——在system prompt中定义Thought/Action/Observation格式，LLM按此格式生成输出。实现要点：(1) few-shot examples展示标准推理-行动-观察循环；(2) action space定义可用工具及其参数格式（如Wikipedia Search["query"]、Wikipedia Lookup["keyword"]）；(3) output parser提取action并路由到对应tool executor；(4) stop condition检测"Final Answer"或达到max iterations。开源实现：https://github.com/ysymyth/ReAct。论文将其适配到vLLM backend：agent worker异步运行，tool执行期间GPU可被其他concurrent request的LLM call填补（inter-request parallelism）。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

