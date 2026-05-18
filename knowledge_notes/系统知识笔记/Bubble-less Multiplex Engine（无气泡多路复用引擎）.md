## Bubble-less Multiplex Engine（无气泡多路复用引擎）

术语是什么？

Bubble-less Multiplex Engine是MuxWise中实现Prefill和Decode在单GPU内高效交替执行的核心调度引擎。其关键设计是将Prefill拆分为小粒度的Block（层级别），在Decode迭代间隙插入执行，消除GPU"气泡"（空闲等待时间）。

论文§3.2.2识别Naive PD multiplexing引入的**三类GPU气泡**：(1) **Kernel Launch不对称气泡**：Prefill launch耗时数十ms，Decode launch（CUDA graph）仅<0.5ms。若先launch prefill再decode，当prefill launch时间长于decode执行时间时，下一个decode iteration无法及时launch，产生bubble。(2) **Decode提前终止气泡**：decode batch中所有请求可能在并发prefill执行期间全部完成，但GPU无法抢占正在执行的prefill kernel，导致SM资源空闲。(3) **长请求导致的SLO Violation**：短对话与长文本摘要并存时，短请求需等待超长prefill完成才能执行，可能错失TTFT SLO。

从系统架构角度拆解术语：

引擎运转流程：

1. **Layer级Prefill拆分**：将一个完整Prefill请求按Transformer Layer切分为多个小Block，每个Block仅处理部分层的计算。
2. **Decode迭代执行**：SLO-aware Dispatcher为Decode分配SM，执行一次Decode迭代（生成一个token）。
3. **Bubble检测与填充**：Decode迭代完成后，引擎立即检查是否有待处理的Prefill Block，将其调度到空闲SM上执行。
4. **持续交替**：每个Decode迭代后都会检查并插入Prefill Block，直至所有Prefill请求完成。
5. **Bubble最小化**：由于Prefill Block粒度小（单层或少量层），即使Decode迭代之间只有很短的间隙，也能被有效填充。

术语一般如何实现？如何使用？

实现上，Prefill被建模为有向无环图（DAG），每个节点为一个Layer的Prefill计算，引擎按拓扑顺序调度。Decode迭代作为高优先级任务，每次迭代完成后释放SM资源，由引擎将等待中的Prefill节点调度上去。这与Chunked Prefill不同：Chunked Prefill按token数切分，PDM按Layer切分以实现更细粒度的调度。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

---
