## Multi-Step Visual Decision-Making

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Step Visual Decision-Making（多步视觉决策）是VLM agent在视觉交互环境中通过多轮观察-动作-反馈循环完成复杂任务的问题范式。与静态VQA（单次图像→文本回答）不同，多步视觉决策要求模型：(1) 在每一步t根据完整历史 H_t = (I, {(o_τ, a_τ, f_τ)}_{τ<t}) 选择动作a_t（I为任务指令，o_τ为视觉观察，a_τ为历史动作，f_τ为环境文本反馈）；(2) 维护跨时间步的内部状态和记忆；(3) 从环境反馈（visual transitions + textual feedback）中推断动作后果和任务进展；(4) 在有限步数内（如20-30步）完成目标。

VisGym将多步视觉决策形式化为RL-style gym paradigm，每个任务的episode由一系列(observation, action, feedback)元组构成，环境在agent发出stop动作时终止并返回二进制成功奖励。核心挑战：VLMs需要将视觉感知（理解图像中的物体位置/状态变化）、长时记忆（跟踪历史交互）和动作规划（选择最优动作序列）三个能力紧密耦合。

从算法pipeline角度拆解术语，给出具体例子。
多步视觉决策的交互循环（以VisGym Jigsaw任务为例）：
```
# Episode execution loop:
Step 0: env.reset() → observation o_0 (scrambled 2x2 jigsaw image)
         prompt = instruction(I) + available_actions + o_0
Step 1: VLM(prompt) → a_1 = ('swap', ((0,0), (0,1)))
         env.step(a_1) → o_1 (new image), f_1 ("Action executed successfully")
         H_1 = {I, (o_0, a_1, f_1)}
Step 2: VLM(H_1 + o_1) → a_2 = ('swap', ((0,1), (1,0)))
         env.step(a_2) → o_2, f_2 ("Action executed successfully")
...
Step k: VLM(H_{k-1} + o_{k-1}) → a_k = ('stop', 'stop')
         env.step(a_k) → reward = 1 if correct else 0
```

关键特性：(a) 每步输入包含完整交互历史；(b) 动作表示为function calls格式；(c) 环境同时提供视觉和文本反馈。

术语一般如何实现？如何使用？
VisGym基于Gymnasium框架实现。评估协议：每环境×setting 70 episodes，easy最大20步/hard最大30步。模型通过OpenRouter API（proprietary）或本地GPU推理（open-weight）。训练协议：使用solver-generated demonstrations进行SFT，仅用easy难度数据训练，hard衡量泛化。关键诊断开关：可控制history length（1/2/4/∞）、observation modality（image/ASCII text）、textual feedback（on/off）、goal observation（with/without）进行受控实验。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---
