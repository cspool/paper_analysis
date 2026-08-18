## Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon

- baseline方法是什么？
  <Baseline 有两类：①**定时器类中断检测**——用架构定时器（cntvct_el0，M1–M3 上 24 MHz）采样时间戳，时间戳跳变超过阈值（如 1μs）即判定发生中断；或自建 counting-thread 定时器（独立线程空循环自增计数器，被中断时计数停摆）。②**loop-counting 攻击**（Cook et al., ISCA 2022）——浏览器内以毫秒级粗粒度定时器采样紧循环完成迭代数，中断使迭代数下降，用中断模式做网站指纹；该攻击依赖 Linux/x86 的**中断亲和（interrupt affinity）**假设：SPI 被绑定到固定单个 core，攻击者只需监听特定 core。Baseline 全栈执行例子（一个受害者访问网站、攻击者做网站指纹）：
  ```
  # 算法层（ML 辅助分类）：LSTM/随机森林按中断 trace 分类网站（本论文 baseline 亦用 LSTM 32 units）
  # 系统框架层：攻击者/受害者用户态进程在 macOS 上运行；中断由网络设备产生并经 Apple 中断控制器（AIC）投递
  # 编译框架层：论文未明确说明（baseline 为用户态 C/JS 代码，无编译框架改动）
  # kernel 调度层（检测原语）：攻击线程在循环里读 cntvct_el0 时间戳 → 差值 >1μs 记为"中断"；
  #   或 counting-thread 循环自增计数器 → 采样间隔内计数偏低记为"中断"；浏览器则数固定间隔内迭代次数
  # 硬件架构层：Apple M 系列 CPU（M1–M4）；baseline 隐含假设与 Linux 相同的 SPI→固定核投递
  # 输出：中断时间序列 → LSTM 分类 → 网站 top-1 准确率（Cook 版 94.8%）
  ```
  痛点：①macOS 不暴露任何中断统计接口（无 /proc/interrupts），用户态也无法绑定 core；②架构定时器可被一系列定时器防御打掉（kperf 已移除、随机定时器注入使 Cook 版准确率降到 ~1%），counting-thread 定时器在中断密集场景下又不可靠（检出率 45.0%–80.5%）且自身增加 active core 数带来额外噪声；③基于 Linux 中断亲和的"监听固定核"假设在 Apple silicon 上是否成立无人知晓，苹果中断控制器 AIC 闭源。>
- 论文方法是什么？如何对应解决Baseline的缺陷？
  <论文方法：**TIDE（Timer-less Interrupt Detection）**——无定时器中断检测技术。关键洞察：Apple 的 Double Map 缓解（Meltdown 防护）在每次 user→kernel 转换时把 AAPCS64 平台寄存器 x18 当作 scratch 寄存器使用（XNU 异常入口的 MAP_KERNEL 宏用 x18 做 mrs/orr/msr TTBR0_EL1 与 msr TCR_EL1 的操作数，BRANCH_TO_KVA_VECTOR 宏用 x18 遍历 TPIDR_EL1→per-CPU→CPU_EXC_VECTORS 表做间接跳转），覆写了用户原始 x18；而 macOS 为防内核信息泄露又在 kernel→user 返回前显式清零 x18。攻击者给 x18 赋非零值，循环检查 x18 是否被清零即可精确判定本核是否发生中断——与定时器完全无关。由此构建两个原语：①中断计时（x18 赋非零后同线程递增计数器直到 x18 清零，用计数器表示相邻中断间隔）；②去噪（测量前后各一次 x18 写/读，清零则丢弃该测量）。再用 TIDE 反推 AIC：SPI 被**均匀投递到所有 active core、忽略 idle core**（与 Linux 的固定核亲和相反）。对应解决 baseline 缺陷：①定时器防御失效——TIDE 不依赖任何定时器，随机定时器注入对其无效；②counting-thread 不可靠——TIDE 精确无噪（与架构定时器 >99% 一致且排除其 0.2%–0.5% 误报）；③core 假设问题——反推出"所有 active core 都收 SPI"后，攻击无需 core 绑定即可捕获 victim 中断（图 6：SPI 只投 active core 也顺带封死 Mwait/IdleLeak 类攻击）；④ML 训练失配——loop-counting 训练标签加入 active core 数（类别=网站×核数），使分类器对额外线程导致的核数变化鲁棒（94.8%→噪声下 39.3%→增强后 92.8%/93.6%）。论文方法全栈执行例子（同一网站指纹场景）：
  ```
  # 算法层：LSTM(32 units) 分类不变，但输入为 TIDE trace（相邻中断间隔序列）；loop-counting 变体在训练标签中并入 active core 数
  # 系统框架层：攻击线程设 x18=0x1 进入检测循环；victim 访问网站产生网络 SPI，AIC 将其均匀投到所有 active core
  # 编译框架层：论文未明确说明（无编译框架改动；需 XNU 内核知识定位 x18 清零行为）
  # kernel 调度层（检测原语）：循环中读 x18，非零则自增计数器，检测到 x18==0（macOS 在 kernel→user 返回前清零）则跳出记录计数器
  #   值 = 中断间隔；去噪：测量代码前后各一次 x18 读/写，清零则丢弃该测量；无任何定时器参与
  # 硬件架构层：Apple M 系列 CPU；AIC 将 SPI 均匀投递到所有 active core、忽略 idle core（反推结论，
  #   非 Linux 固定核亲和）；因此攻击线程落任意 active core 都能收到 victim 的中断
  # 输出：TIDE trace → LSTM → 网站 top-1 93.8%（closed-world）/91.5%（open-world）/视频 78.1%；
  #   SysBumps 去噪后 54%→81%；隐蔽信道 111.65 bps；SIKE 密钥提取成功
  ```
  （全栈中"模型推理/算法"为攻击侧的 ML 分类器而非推理系统，"系统框架/kernel 调度/硬件"均为 macOS/Apple Silicon 真机运行时，无 Serving 框架、编译器、RTL 模拟器或芯片设计层次。）>
