## Sound Spatialization（声音空间化 / SS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 声音空间化（Sound Spatialization, SS）是把场景中的源音频转换成听者双耳（binaural）信号的过程，显式建模声音在环境中如何传播、如何受听者位置与朝向影响，是 VR 沉浸感与空间感知的关键（区别于普通音频渲染的解码/混音/panning/简单混响）。ECHO 论文把 SS 流水线分解为三阶段（对应其 Fig.4）：①声传播（Sound Propagation）——用镜像源法（ISM, Image Source Method）模拟声音在室内经墙面/障碍物反射、衍射、混响后到达听者的传输路径，基于听者位姿、源位置、场景几何与材质计算房间冲激响应（RIR）；②BRIR 生成（Binaural Room Impulse Response Generation）——用听者特定或通用的头相关传递函数（HRTF, Head-Related Transfer Function）把 RIR 转换成左右耳各自的双耳房间冲激响应（BRIR），建模头、躯干、耳朵的滤波；③可听化（Auralization）——把源音频与 BRIR 做卷积生成空间化信号。三个源在不同位置时三阶段需独立执行，因此渲染成本随源数线性上升。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - SS 在 VR 中是一个感知-估计-渲染闭环流水线：传感（S^M 单目 SLAM 图像 + S^I IMU）→ 位姿估计（PE）→ 声传播+BRIR 生成（R^1）→ 可听化（R^2）→ DAC 输出。ECHO 论文按此建模 motion-to-sound 延迟 T_m-s = T_IN + T_S + T_P + T_R1 + T_R2 + T_O。具体计算过程（一个声源）：设源在房间位置 x_s、听者位姿 (R,t)，ISM 把源对每面墙镜像得到镜像源 x_s'，对每个镜像源/直达声计算到达听者的路径、延迟 τ 与幅度 a（含墙面吸收），叠加所有路径得 RIR h(t)=Σ_j a_j δ(t-τ_j)（+扩散混响尾）；BRIR 用 HRTF 卷积：h_L(t)=h(t)*hrtf_L(θ,φ)、h_R(t)=h(t)*hrtf_R(θ,φ)，其中 (θ,φ) 是源相对听者的方位/仰角；可听化输出 y_L(t)=s(t)*h_L(t)、y_R(t)=s(t)*h_R(t)。多源时每源重复上述三阶段再求和。ECHO 评估中 R^1 用 Pyroomacoustics（CPU）与 gpuRIR（GPU）的 ISM 实现，R^2 为 BRIR 卷积。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：房间声学模拟库（Pyroomacoustics：Python 房间模拟 + 阵列处理，ISM 生成 RIR/BRIR；gpuRIR：GPU 加速 RIR 模拟）；HRTF 数据集（KEMAR dummy-head 测量的远场 HRTF，或近场 HRTF 数据库）；可听化可用分区卷积（partitioned convolution）做实时长滤波器卷积。在 VR 系统中把最新 head pose 传给渲染器，对每音频块（5-20ms）用最新位姿执行传播/BRIR 生成，再用最近 BRIR 卷积当前块，两阶段异步流水以提升稳态吞吐（ECHO Fig.6b）。实时性约束：motion-to-sound 延迟须 <50-60ms 保沉浸感。优化方向（ECHO）：按头朝向做声学注视（acoustic foveation）减少活跃源数、用 GPU 并行（gpuRIR）加速传播、算法-硬件协同。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
