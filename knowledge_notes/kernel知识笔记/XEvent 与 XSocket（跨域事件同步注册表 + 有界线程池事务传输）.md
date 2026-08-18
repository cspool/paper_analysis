## XEvent 与 XSocket（跨域事件同步注册表 + 有界线程池事务传输）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XEvent 是 UCV 透明软硬件映射层的事件同步机制：通过 Event Registry 把 HVL（SystemVerilog/UVM）事件映射为软件事件对象——(i) 以字符串标识符注册跨语言事件、(ii) 镜像事件状态与参数缓冲区、(iii) 跨语言代理回调。XSocket 是事务传输机制：把 TLM 风格事务 transport 派发到有界线程池，将同步阻塞等待转换为异步阻塞，避免"模拟器等软件线程、软件等模拟器推进时间"的相互等待死锁；硬件侧保留传统 TLM 实现，软件侧暴露 socket 风格 API。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
事件同步流程（Fig.5）：
```
await HVL事件 → 注册表登记字符串标识符+参数缓冲
→ HVL trigger 更新注册表事件状态与参数
→ 同步点（Fig.4 数据流处）跨环境传播更新，恢复调度器可见性
# 无时序场景：注册表按标识符解析目标函数，经函数指针+绑定层解码同步派发（动态跨语言调用）
```
事务传输死锁与规避（Fig.6）：若软件扩展在步骤 2/3 强制软硬件异步上下文切换，先前的线程上下文不保留 → HW Transport 无法把控制权交还调度器，与模拟器形成相互等待死锁；XSocket 让 transport 代码运行在有界 worker 线程上、调度器在步骤 7 投递异步通知 → 模拟器不再等待软件线程，同时线程开销有界。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
性能（本文 Table IV，NoC 13,036 LOC / ICache 5,163 LOC）：UCV+（UCV 启用 UVM 支持）执行快约 16.6%、验证代码量少 12%（NoC：24.06s/15.41s vs UVM 15.32s/18.47s；ICache：13.69s/94.36s vs 19.14s/106.12s）——UVM 流程集成软件激励与 VIP 需进程间通信（共享内存等待/同步/序列化），XSocket 换为进程内直接传输后消除该开销。典型使用：软件测试任务与模拟器内 UVM VIP 协调（BPU 验证 Step 3），VIP 事件经 XEvent 注册表镜像、事务经 XSocket 线程池异步化。Web 无外部来源（UCV 平台特有机制）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization
