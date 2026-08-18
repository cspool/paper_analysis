## XClock 与 XData（软件原生时序模型：软件声明时钟 + 边沿对齐数据）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XClock 是 UCV 定义的软件声明时钟：host 与模拟器共享的规范时间基（频率/时钟边沿/相位），由软件（而非模拟器）规定 commit/sample 时间点并约束模拟器何时可推进、I/O 何时有效；它实现为从既有事件循环（asyncio、Boost.Asio）调用的普通库，不改模拟器/RTL/第三方库。XData 是绑定 XClock 的时序感知数据类型，双层抽象：下层 C/C++ 提供模拟器控制与不含时序语义的信号访问（经 SWIG 绑定暴露给 HLL），上层用语言原生并发模型做事件编排与边沿对齐——软件不用为每次读/写标注显式边沿等待。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
软件迭代循环（本文 Fig.4 的软硬件交互调度）：
```
loop:                          # 每轮软件迭代
  1. 处理 T0 时刻全部 pending 软件事件
  2. 缓冲写提交到模拟器（T0 边沿 commit）
  3. HWStep(T1 >= T0)          # 控制权进入模拟器
  4. 返回后按需读信号；值/时间变化的回调派发新事件
  5. 新事件入队，继续循环
# HWStep 内部：每个周期处理当前时刻全部事件 → 时间+1 → 直到 T0==T1 返回；
# T1==T0 时只推进零时相位至静止（观察组合逻辑与 δ-cycle）
```
XData 默认只暴露时钟边沿点（上升沿，可选下降沿）作为传输调度点，可声明额外边沿做周期内细观测。该调度同时具备：事件级表达力（重叠 in-flight 交互各占独立异步流）+ 周期精确语义（边沿对齐 commit、静止后采样）——对比之下 step-peek 只有整周期步进、cocotb 回调可能早于信号稳定触发。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
API（本文 Table II）：XClock(pin,thigh,tlow)、XEvent(conds,args,sclk,reactors)、XEvent(conds,sclk).await、XTrigger(events/reactors,sclk)、XReactor(name,cb,sclk)、@XReactor 装饰器。应用形态：Python asyncio 中 await 时序点推进、pytest 编写测试；跨语言（C++/Python/Java/Go）经 SWIG 绑定同一 backend adapter。开销数据（Fig.13）：XiangShan（3.45M LOC）吞吐损失 ≤3%（大设计时间几乎都在模拟器执行）；CoupledL2/RocketChip 损失 14%–55%（小设计每秒周期数高、软件事件调度占比大）；峰值内存主要随语言变化（JVM/解释器开销）。Web 无外部来源（UCV 平台特有机制，见 https://github.com/XS-MLVP/picker 生态）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization
