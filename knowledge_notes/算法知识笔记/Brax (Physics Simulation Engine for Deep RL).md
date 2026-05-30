## Brax (Physics Simulation Engine for Deep RL)

术语是什么？
Brax 是 Google 开发的基于 JAX 的可微分物理仿真引擎，专门用于大规模刚体仿真以支持深度强化学习（Deep RL）训练。Brax 将物理仿真（刚体碰撞检测、接触力计算、关节约束求解等）映射为 GPU kernel，通过 JAX 的 XLA 编译器生成 CUDA kernel 在 GPU 上并行执行。Brax 支持多种仿真环境（Ant、Humanoid、Grasp、Cheetah、Walker2d 等 MuJoCo 兼容环境），每个环境包含：刚体定义（link 质量、惯性、几何形状）、关节约束（joint 类型、运动范围）、actuator 定义（驱动力矩）、碰撞检测和接触动力学。在 Deep RL 训练中，Brax 用于数据采集阶段——agent 的策略网络（DNN）根据当前环境状态输出动作，Brax 在 GPU 上并行仿真多个环境实例，产生下一状态和奖励。

从算法pipeline角度拆解术语：
Brax 中一次 Deep RL 训练步骤的 pipeline：
```
Algorithm: Deep RL Training Step with Brax
Input: policy_net (DNN), env (Brax simulation environment)
Output: training_batch

// Phase 1: Data Collection (在GPU上)
// 并行模拟 N=4096 个环境实例
states = env.reset(batch_size=4096)   // (4096, state_dim) on GPU

for step in range(rollout_length):     // 如 rollout_length=10
    // 1. 策略推理: 所有环境实例并行
    actions = policy_net(states)        // 大量小kernel → GPU occupancy低
    
    // 2. 物理仿真步进: 每步包含数百个小kernel
    //    - collision_detection: 检测刚体对之间的碰撞
    //    - contact_forces: 计算接触力
    //    - joint_forces: 计算关节约束力
    //    - forward_dynamics: 从力计算加速度
    //    - integrate: 更新位置和速度
    next_states = env.step(states, actions)  
    // 小kernel问题: 每次碰撞检测根据input不同有不同的kernel路径
    // (哪些刚体对接触取决于当前状态 → input-dependent graph)
    
    states = next_states
    trajectory.append((states, actions, rewards))

// Phase 2: Policy Update (在GPU上)
// 从trajectory采样batch训练policy_net
loss = policy_update(trajectory)  // 标准DNN训练，计算量通常可饱含GPU
```

数据采集阶段占 Deep RL 训练时间的 30-70%（取决于环境复杂度），而该阶段 GPU occupancy 仅约 34%（平均），原因正是 Brax 产生的大量小 kernel——每个仿真步骤需要数百个 kernel launch，每个 kernel 仅有少量 CTA（中位数 < 200 CTA），无法填满 GPU（RTX 3060 28 SM）。

术语一般如何实现？如何使用？
Brax 开源（github.com/google/brax），基于 JAX（jax.readthedocs.io）。安装：`pip install brax`。使用流程：(1) 选择/自定义仿真环境（`brax.envs.create("ant")`）；(2) 调用 `env.reset()` 初始化状态；(3) 调用 `env.step(actions)` 推进仿真；(4) JAX 的 `vmap` 自动并行化多个环境实例。Brax 训练 pipeline 集成在 `brax.training` 中，包括 PPO、SAC 等 RL 算法。ACS 论文使用 Brax 的 5 个 MuJoCo 环境（Ant, Grasp, Humanoid, Cheetah, Walker2d）评估 ACS 的效果，在 ACS-HW 下实现 Deep RL 训练端到端加速 1.42×（平均）。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
