## AI Agent / Dynamic Reasoning（AI智能体/动态推理）

术语是什么？通过联网搜索让回答具体和精准。

AI Agent是基于LLM的推理时框架，通过多步推理、自适应决策和与外部环境的交互来扩展LLM的能力。与传统的单轮LLM推理（输入→静态prompt→输出）不同，AI Agent执行动态推理（Dynamic Reasoning）：在每个iteration中，agent可能生成中间推理结果、调用外部工具（如搜索引擎、计算器、代码解释器），并将工具返回结果纳入后续决策中。这个过程允许agent动态获取缺失信息、根据任务需求调整策略。AI Agent通常包含四个核心组件：(1) Agent Core：由LLM担任actor/planner/reflection等角色；(2) Memory：存储短期交互轨迹和长期知识；(3) Plan：将目标分解为子任务序列或DAG；(4) Tools：扩展能力的外部接口（如Wikipedia API、Wolfram Alpha、Python executor等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

AI Agent的算法pipeline是LLM inference和tool execution交替迭代的过程：

```
// AI Agent 单请求执行pipeline
function AgentExecute(query, workflow_type):
    context = [Instruction, FewShot, Query]
    history = []
    
    while not termination_condition():
        // Phase 1: LLM Inference
        prompt = concat(context, history)
        llm_output = LLM_Backend.generate(prompt)  // prefill + decode
        
        if llm_output.is_final_answer():
            return llm_output
        
        // Phase 2: Parse Action
        action = parse_action(llm_output)  // 解析thought/action
        
        // Phase 3: Tool Execution
        tool_result = Tool_Executor.execute(action.tool, action.params)
        // GPU可能在tool执行期间idle
        
        // Phase 4: Update Context
        history.append((llm_output, tool_result))
```

不同agent workflow的区别在于termination condition和action空间：
- **CoT**：无tool调用，仅内部reasoning，单次LLM inference
- **ReAct**：交替reasoning和tool use，直到达到final answer或iteration limit
- **Reflexion**：在ReAct基础上周期性插入self-evaluation和refinement
- **LATS**：Monte Carlo Tree Search，每个tree node展开时发出多个并行LLM calls评估候选path
- **LLMCompiler**：先用planner构造DAG任务依赖，再streaming async执行tool calls

论文测量：tool-augmented agent平均LLM calls是CoT的9.2倍，LATS平均71.0次LLM calls/request。LLM inference和tool execution分别占总延迟约69.4%和30.2%，但由于LLM输出决定下一步调用哪个tool，两者难以重叠（LLMCompiler的DAG并行仅占总延迟18.2% overlap）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AI Agent通过agent framework实现：将LLM backend（vLLM/OpenAI API等）与tool system连接。常见实现方式：(1) agent worker（Python async process）维护state machine，根据workflow决定下一步是LLM call还是tool call；(2) LLM backend负责prefill+decode；(3) tool system执行本地代码（Python interpreter）或外部API（Wikipedia/Wolfram Alpha/web browsing）；(4) 开源框架包括LangChain、AutoGen、CAMEL等。论文使用各agent原作者的开源实现，统一整合到AgentBench框架（https://github.com/VIA-Research/AgentBench）：ReAct (github.com/ysymyth/ReAct)、Reflexion (github.com/noahshinn/reflexion)、LATS、LLMCompiler (github.com/SqueezeAILab/LLMCompiler)，统一适配到vLLM backend（vLLM 0.6.6, PyTorch 2.6, CUDA 12.8）。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

