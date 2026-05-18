## LLMCompiler (Structured Planning for Parallel Function Calling)

术语是什么？通过联网搜索让回答具体和精准。

LLMCompiler是一种基于结构化多步规划的AI agent workflow。与ReAct的交替thought-action循环不同，LLMCompiler先用planner分析任务依赖并构造一个DAG（有向无环图），将多个tool calls组织成可执行计划。计划生成过程中，中间tool calls可以streaming到execution stage，让scheduler异步执行工具，从而将部分planning和tool execution overlap，降低端到端延迟。这种"plan-then-execute"范式减少了整体LLM call次数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// LLMCompiler执行pipeline
function LLMCompiler_Execute(query):
    // Phase 1: Planning (单次LLM call)
    plan_dag = Planner_LLM(query)  
    // 输出: DAG nodes = tool calls with dependencies, edges = precedence
    
    // Phase 2: Streaming Execution (与planning重叠)
    scheduled_tasks = []
    for node in plan_dag.topological_order():
        if node.dependencies_met():
            task = async_execute_tool(node.tool, node.params)
            scheduled_tasks.append(task)
    
    // Phase 3: Join results
    results = await gather(scheduled_tasks)
    
    // Phase 4: Final answer generation (第二次LLM call)
    final_context = [plan_dag, results]
    return LLM(final_context)
```

论文测量：LLMCompiler的planning-tool overlap仅占总延迟约18.2%（受任务依赖关系限制）。在HotpotQA上LLMCompiler在accuracy和cost-efficiency上均优于ReAct（结构化规划减少重复推理）。但在WebShop上因tool使用涉及高互依赖（搜索→点击→翻页），DAG-style planning导致不必要的tool invocations，效率低于ReAct。LLMCompiler不适合MATH/HumanEval等需要强顺序推理的任务。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LLMCompiler核心组件：(1) Planner LLM：分析任务、识别tool依赖、生成DAG计划（单次LLM call生成完整计划）；(2) Task Fetching Unit：按DAG topological order stream task到execution stage；(3) Executor：异步执行tool calls（支持并行执行无依赖tasks）；(4) Joiner：收集tool results并format成final LLM prompt。论文将其适配到AgentBench统一框架，使用vLLM backend。开源实现：https://github.com/SqueezeAILab/LLMCompiler。LLMCompiler的关键优势是单次planning LLM call生成完整计划，减少整体LLM call次数和重复推理。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

