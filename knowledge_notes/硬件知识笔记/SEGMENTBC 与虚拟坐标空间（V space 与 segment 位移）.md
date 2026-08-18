## SEGMENTBC 与虚拟坐标空间（V space 与 segment 位移）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SEGMENTBC 是 SegFold 的 SpGEMM 动态映射机制，负责把 B 元素按列索引"即时"定位/创建其在输出 C 中的约简位置。由于 C 与中间张量 T 的稀疏结构高度不规则且数据依赖，SegFold 不物化中间张量，而是维护一个压缩的虚拟坐标空间 V=X×Y：|X| 为非空 C 行数、|Y| 为任一 C 行的最大非零数，每个被占用的虚拟坐标 (x,y) 持有一个不同的 C 元素（部分和）。映射 f_t(m,n)=(x,y) 把稠密笛卡尔坐标 (m,n) 映射到 V space 的虚拟位置，受四条约束：injectivity（不同 (m,n) 不同坐标）、row saturation（虚拟行内非零从左到右连续无空隙）、column ordering（虚拟行内列索引严格递增）、time ascending（条目只随 t 前移）。segment 定义为 B 元素从注入点 f_tin 到最终消费位置 f_tout 的位移；SEGMENTBC 的目标是最小化 segment 位移以降低网络争用。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
V space 更新例子（论文 Fig.4）：某 C 虚拟行在 t 时刻存列索引 {0,2}（底部行）；新到达的 B 元素与匹配的 A 生成列索引 {0,1} 的部分和。因列索引 1 此前未出现，为满足 column ordering，映射更新为 {0,1,2}（插入空位、已有条目右移）。硬件上每个 PE 存一个虚拟坐标位置（PE r 行 p 列 = V 坐标 (r,p)），PE 行表示一条 C 虚拟行、按列索引升序排列；B 元素经 IPM 定注入点后进入 merge network，merger 比较 b 与 c：b>c 向右转发、b<c 触发右侧整体右移插入空位、b=c 就地累加。segment 位移公式：displacement = ||f_tin(m,n) − f_tout(m,n)||，t_in 为进入阵列时刻、t_out 为消费时刻。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SEGMENTBC 逻辑由自适应 merge network（每 PE 一个 merger）+ Index-to-PE Mapper（IPM，二叉搜索定注入点）+ 每 PE 行本地 spad（存溢出 C 值）+ 本地 IPM 查找表组成；PE 行有专用连接共享这些小内存，访问受限避免争用。使用：作为 Segment 数据流的执行引擎，它使 C 元素能在 PE 间动态迁移（区别于静态输出驻留），把约简负载按运行期 V space 状态再平衡；映射消融实验（LUT-based vs zero-offset vs ideal-oracle）显示 LUT 映射比 zero-offset 快 1.20× geomean、相对理想 oracle 仅 1.2% 平均开销。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
