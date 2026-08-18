## OpenMP / std::thread / Fortran DO CONCURRENT（语言级并行模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
三种语言/编译器层级的并行编程模型，SPEC CPU2026 的 22 个并行 SPECspeed benchmark 全部采用其中之一（另加 task-based 进程 spawn）。OpenMP 3.0：基于编译制导指令（#pragma omp parallel for 等）+ 运行时库的共享内存并行标准，支持线程派生/归并、worksharing、同步与 task（OpenMP 3.0 引入 task 构造）；C++ std::thread：C++11 标准线程库，程序显式创建/join 线程对象（与 OpenMP 的隐式线程池不同）；Fortran DO CONCURRENT：Fortran 2008 引入的循环构造（DO CONCURRENT (i=1,n) ... END DO），声明迭代间无依赖、允许编译器自动并行化/向量化，是 Fortran 侧"编译器可证明安全并行"的写法。CPU2026 语言标准基线：C18（从 C99）、C++17（从 C++03）、Fortran 2018（从 Fortran 2003）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
三种模型对应三种"并行性表达 → 运行时执行"路径：
```
OpenMP: #pragma omp parallel for → 编译器生成 GOMP/libgomp 运行时调用
  → 线程池创建 N 线程 → 迭代静态/动态/guided 分块派发 → barrier 归并
  (SPEC 例子: 722.palm 大气模拟、709.cactus 相对论有限差分等 OpenMP MT benchmark)
std::thread: std::thread t(f, args) → 编译器直接映射 pthread_create/内核线程
  → 显式 join 同步 (无隐式线程池, 无制导指令)
DO CONCURRENT: DO CONCURRENT 循环 → 编译器分析无跨迭代依赖 → 自动并行化/向量化
  → 生成线程化或 SIMD 代码 (或串行, 取决于编译器能力)
task-based: 821.gcc/823.llvm 逐条 spawn 编译器进程 (make -j N 语义) —— 进程级并行
```
SPEC CPU2026 用这些模型覆盖从"编译制导"到"显式线程"到"声明式并行循环"到"进程级任务"的并行实现谱系；论文附录用 128 线程跑 SPECspeed 做 IPC/瓶颈特征化，并指出 800.pot3d/801.xz 呈现最多锁争用（鼓励研究线程扩展性与数据共享）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：OpenMP 由编译器（GCC libgomp、LLVM libomp）翻译制导指令并链接运行时；std::thread 由标准库/系统线程 API 实现（Linux 上 pthread）；DO CONCURRENT 是 Fortran 编译器前端语义（-fopenmp 可让 GCC 把 DO CONCURRENT 并行化，gfortran 在 -O2/-O3 下可自动并行）。SPEC 的可移植性工程与此直接相关：为消除"threading tax"，单线程 SPECrate 版把多线程应用的锁/互斥同步抑制掉，同时保留必要 hooks 供多线程 SPECspeed 版使用（§III-B Suppression of Threading Artifacts）；TSan 检测多线程 benchmark 的数据竞争（867.nest/837.gmsh）。使用场景：CPU2026 的 SPECspeed MT benchmark 用于强扩展/多核性能评估、锁争用与缓存一致性研究；三种模型的并存也使套件覆盖不同并行实现的编译/运行时开销特征。

涉及论文标题：
- SPEC CPU: The Next Generation
