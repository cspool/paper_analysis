## Gymnasium (RL Environment Framework)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gymnasium是OpenAI Gym的后继开源强化学习环境框架，提供统一的`Env`基类和标准化的`reset()`/`step()`/`render()`接口用于RL agent训练和评估。Brockman et al. (2016)首创Gym接口，Towers et al. (2024)将其维护为Gymnasium项目。MuJoCo和Atari等经典RL benchmark均基于此框架。VisGym在Gymnasium基础上扩展了四项增强功能：(1) Function-Conditioned Action Space——动作表示为带参数的函数调用；(2) Function Instructions——每个任务的自然语言函数说明；(3) Environment Feedback——除视觉变化外的文本反馈；(4) Solver——内置启发式求解器生成demonstration。

从系统架构角度拆解术语，给出具体例子。
Gymnasium的核心接口和VisGym扩展：
```
# 标准Gymnasium接口
env = gymnasium.make("TaskName-v0")
obs, info = env.reset()
for step in range(max_steps):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# VisGym扩展接口
env = visgym.make("Jigsaw-v0", difficulty="easy")
obs = env.reset()  # returns PIL Image observation
history = [(obs, None, None)]  # (o_τ, a_τ, f_τ)
for step in range(max_steps):
    prompt = build_prompt(instruction, available_actions, history)
    action_str = vlm.generate(prompt)  # e.g., "('swap', ((0,0), (0,1)))"
    obs, reward, terminated, truncated, feedback = env.step(action_str)
    history.append((obs, action_str, feedback))
```

VisGym的Step函数（Algorithm 1）处理流程：解析action string → 验证format + action name + payload → Apply执行 → 返回(obs, reward, terminated, truncated, feedback)。所有17个任务共享此统一接口，每个任务通过定义各自的`action_space()`和`Apply()`实现差异化。

术语一般如何实现？如何使用？
安装：`pip install gymnasium` + `pip install visgym`（或从GitHub clone）。VisGym提供Python API创建和交互环境。评估集成：通过OpenRouter API调用proprietary模型，或本地加载open-weight模型。训练集成：通过LlamaFactory处理demonstration数据和SFT训练编排。Gymnasium的标准化接口使VisGym环境可被任何支持RL-style交互的pipeline消费。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---
