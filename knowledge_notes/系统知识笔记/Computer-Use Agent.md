## Computer-Use Agent

术语是什么？

Computer-Use Agent 是让LLM通过操作计算机界面（GUI或API）完成复杂任务的AI Agent范式。与传统chatbot（单次inference返回结果）不同，Computer-Use Agent需要多轮交互：观察当前屏幕/应用状态→规划子步骤→执行操作（点击、输入、拖拽、API调用）→观察反馈→迭代直至完成任务。代表性框架包括Microsoft UFO（基于Windows UIA的GUI agent）、OSWorld（benchmark）、Claude Computer Use、OpenAI Operator等。Computer-Use Agent的交互方式分为两类：GUI路线（通用但成功率低，受LLM视觉定位和精确操作能力限制）和API路线（成功率高但仅适用于有公开API的应用）。

从系统架构角度拆解术语：

典型的Computer-Use Agent系统架构（以UFO-2 + DMI为例）：

```
[User Task] → [Agent Orchestrator] → [LLM (policy)]
                    ↕
            [Execution Layer] → [OS Interface]
              /        \
      [DMI fast-path]  [GUI slow-path]
              ↓              ↓
      [UIA Control]   [Screen coordinate + UIA labels]
```

1. **Agent Orchestrator**：接收用户任务，管理LLM调用的生命周期——构造prompt（含任务描述 + 应用状态 + 可用操作接口）→调用LLM→解析LLM输出→分发给Execution Layer→收集执行结果→更新上下文→决定继续或终止。
2. **LLM (Policy)**：负责语义理解和规划——将自然语言任务分解为子步骤，根据当前状态决定下一步动作。不直接执行操作。
3. **Execution Layer**：负责将LLM的policy决策转化为实际OS操作。DMI fast-path通过声明式接口确定性执行；GUI slow-path通过屏幕坐标/UIA label点击执行。
4. **OS Interface**：提供应用状态的底层访问能力，包括UIA tree、screenshot、process management等。

术语一般如何实现？如何使用？

Computer-Use Agent一般通过task benchmark评估：(1) OSWorld——跨应用操作系统级任务benchmark（Ubuntu虚拟机上的真实GUI操作），论文使用其Windows变体OSWorld-W中27个单应用Office任务；(2) Agent success rate (SR)为主要指标——任务是否在规定步数内完成；(3) 失败分析将失败分为policy-level（LLM语义理解/规划错误）和mechanism-level（控件定位/导航/交互错误）。当前SOTA DMI+GPT-5 medium在Office任务上SR为74.1%（vs GUI-only 44.4%），失败81.0%为policy-level。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---
