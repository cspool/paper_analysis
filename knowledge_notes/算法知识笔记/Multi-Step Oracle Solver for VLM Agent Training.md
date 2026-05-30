## Multi-Step Oracle Solver for VLM Agent Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Step Oracle Solver（多步神谕求解器）是VisGym中为每个视觉交互环境设计的启发式算法，能使用环境提供的action functions完整完成任务。Solver扮演双重角色：(1) 验证任务实例可解性；(2) 生成demonstration轨迹用于VLM的监督微调（SFT）。每个solver支持多策略（如Jigsaw的swap/reorder策略）和可选随机性（通过插入可逆padding动作对生成多样化轨迹）。

Solver使用经典算法：BFS（Sliding Block、Matchstick Equation）、DFS（Matchstick Equation）、图搜索（Maze 2D/3D）、状态机oracle（MuJoCo Fetch）、贪心重排（Jigsaw/Video Unshuffle）。

从算法pipeline角度拆解术语，给出具体例子。
Maze 2D solver生成demonstration的流程：
```
1. GraphSearch(maze_grid, start, target):
     shortest_path = BFS from start to target
     action_seq = convert to move(0/1/2/3 for right/up/left/down)
2. Optional padding:
     for i in range(target_steps - len(action_seq)):
         insert reversible pair: move(d), move(opposite(d))
3. Return trajectory: [(o_0,a_1,f_1), (o_1,a_2,f_2), ..., stop]
```

其他solver设计（Appendix A）：
- **Sliding Block**: BFS最短序列 → pad with back-and-forth move pairs
- **Matchstick Equation**: BFS/DFS → SOS策略（最短路径 + 随机可逆detours）
- **Jigsaw**: reorder（单次排列整个puzzle）或swap（贪心逐个纠正）
- **Fetch Pick-and-Place**: 状态机oracle（open→descend→close→per-axis move to goal）
- **Mental Rotation 3D**: 分解为yaw/pitch/roll → 每轴4×90° padding旋转后再纠正

术语一般如何实现？如何使用？
Solver作为环境类内置方法实现。Demonstration生成流程：(1) solver生成最优动作序列；(2) replay生成完整(observation, action, feedback)轨迹；(3) 预处理过滤失败轨迹和test-set重叠的初始状态；(4) 仅easy难度demonstration用于SFT训练（hard衡量泛化）。多策略和随机性使同一任务实例可生成多条不同demonstration。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---
