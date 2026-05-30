## Function-Conditioned Action Space

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Function-Conditioned Action Space（函数条件化动作空间）是VisGym为VLM agent设计的一种动作表示方法，将agent的动作建模为带参数的函数调用而非传统离散/连续动作向量。例如，('swap', ((0,0), (0,1)))表示交换两个坐标的拼图块，('rotate', (30.5, 20.4, 15.1))表示绕三个轴旋转。设计动机：VLMs天然具备function-calling能力（经过instruction tuning），使用函数调用格式可充分利用这一能力实现跨任务策略组合。

从算法pipeline角度拆解术语，给出具体例子。
VisGym统一step函数处理function-conditioned action的流程：
```
function Step(a):  # a = "('swap', ((0,0), (0,1)))"
    ρ ← 0; (τ, υ) ← (false, false)
    Parse a → (α, π)  # action name, payload
    if invalid format: return (obs, 0, τ, υ, "invalid format")
    if α in A and π in A[α]:
        (φ, τ, υ) ← Apply(α, π)  # execute, return feedback + flags
    else: return (obs, 0, τ, υ, "invalid action")
    if τ: ρ ← ComputeReward()
    return (obs, ρ, τ, υ, φ)
```

17个环境各自定义可用action function集合（Table 2）：
- Colorization: rotate(θ), saturate(δ), stop()
- Maze 3D: move(0), turn(d), stop()
- Matchstick Equation: move([si, ss, di, ds]), undo(), stop()
- MuJoCo Fetch: move([x,y,z]), gripper(g), stop()
- Jigsaw: swap((r1,c1),(r2,c2)), reorder([...]), stop()

术语一般如何实现？如何使用？
Function-conditioned action space通过Gymnasium扩展接口实现。每个任务的`action_space()`返回可用函数列表和参数约束，`step(action_string)`通过解析器提取函数名和参数。使用方式：(1) 初始prompt包含所有函数的自然语言描述（Function Instructions）；(2) VLM每步输出格式化的函数调用字符串；(3) 环境解析执行后返回新的视觉+文本反馈。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---
