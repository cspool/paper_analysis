## 顺序一致性（Sequential Consistency，SC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SC（Lamport 1979）是最直观、最强的内存一致性模型：要求全局内存序 <m 是所有线程对所有地址全部内存访问的全序（total order），且每个线程在该全序中的访问必须符合其程序序 <p；任何 load 返回全局序中该地址"最近一次"store 的值。若不存在这样的全序——即执行序叠加 MCM 要求的 <p 回边后出现环（cycle，如 QED 图 1(b) 的乱序 load 造成环）——则系统违反 SC。SC 的传递性推论（QED III-A）：只需保持相邻指令对（consecutive pair）的程序序，a<p b<p c 中保持 a<p b 与 b<p c 即可经传递性保证 a<p c，无需单独检查 a-c 对——这是 QED "直接序对定理"对 SC 的直观例证。Web 佐证：Lamport《How to make a multiprocessor computer that correctly executes multiprocess programs》（1979）；QED 引用 [35]。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
QED 对 SC 的验证例子（ld-ld 对不同地址，图 7/9）：主线程 ld A 与 ld B 乱序（ld B 先执行），引入两个 invalidation inv B/inv A 作为外部 store 的代理，枚举 inv B 与 ld B 的 <m 序（inv B<m ld B 或 ld B<m inv B）等全部组合；含环的 trace（如 ld B<m inv B<m inv A<m ld A 叠加 <p 回边）即 SC 违反的反例（对应 ld A=1、ld B=0），无环者（inv B<m ld B）SC 允许（ld A=1、ld B=1）。QED 为 SC 自动生成 26 棵探索树（区分同/异地址，图 9 的 ld-ld 树），其中 4 棵 trivial、38 个谓词（表 II）——远小于现代指令窗口的重排组合。SC 只有 load/store 两类指令参与排序。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现 SC 代价高：load 不得早于更老 store 完成，商业处理器多采用 TSO/弱序模型换取性能；实现上常靠 store buffer + 一致性探针把 load 归位。验证侧 QED 用"探索树 + 环检测 + 谓词决策树"对 RTL 无界验证；QED 自动发现的反例正是教科书经典场景（图 1(b)），说明该方法是穷举而非手工构造。与 TSO/RVWMO 的关系见"内存一致性模型（MCM）与 RVWMO"、"TSO 与 Release Consistency"条目。

涉及论文标题：
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
