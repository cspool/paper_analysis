# <span id="page-6-5"></span>5 Task Fusion Optimizations

Having described our algorithm for task fusion, we now describe optimizations necessary for a practical implementation. We show how to eliminate temporary distributed data structures (Section 5.1) and how to memoize the fusion analysis (Section 5.2). Temporary elimination and memoization are widely applied optimizations; we discuss how to perform these optimizations in a distributed, task-based setting.

#### <span id="page-6-0"></span>5.1 Temporary Store Elimination

Once Diffuse identifies a fusible prefix of tasks, stores that fusion has made temporary may be promoted into task-local data. Conversion of distributed data into task-local data is critical for realizing the benefits of fusion, as task-local data can then be optimized away (Section 6) to maximize reuse.

To introduce when a store is temporary, consider the cuPyNumeric program in Figure 6a and the resulting task stream in Figure 6b. This example introduces some new operations, specifically flush\_window, which sends all pending tasks through Diffuse to the underlying runtime system, and the Python del operator, which drops references. The program creates the stores x, y, z, w, and v. Consider the program state after line 10: the tasks that initialize x and y have executed, as the first flush\_window call sent those tasks to Diffuse. We note that there are no pending tasks outside the window, and future tasks are ones the application may

launch once the call to flush\_window returns. The fusion algorithm determines that the tasks issued by lines 4–6 can be fused, while the final norm must be excluded. First, v is not temporary because the application holds a reference to it, meaning that it could launch a task that reads v after the call to flush\_window(). Next, while the application has deleted its reference to w, the norm task reads a piece of w and is still pending after the fused task, and thus must observe any effects performed on w, meaning that w is not temporary. The stores x and y are only read by the fused task, and thus are not temporary. Only z is temporary because it is produced entirely within the fused task and is not visible to the application or pending tasks. We formalize this intuition as constraints that must be satisfied for a store to be temporary.

**Definition 4.** Given tasks  $[T_1, ..., T_f, ..., T_n]$ , a store S is *temporary* in the fusion of  $[T_1, ..., T_f]$  if

```
1. If \exists T_j, P \text{ s.t. } R(T_j, (S, P)), \exists T_i \text{ such that } i < j \land W(T_i, (S, P)) \land \text{covers}(S, P)
```

- 2.  $\not\equiv T_k, P \text{ s.t. } k > f \land \mathsf{R}(T_k, (S, P)) \lor \mathsf{Rd}(T_k, (S, P))$
- 3. *S* has no live application references.

The function covers(S, P) is true when the partition P contains all points in the store S. The first two constraints check that the store's contents are entirely created within the fused task and not used by any other existing task; these conditions are checked through a forwards dataflow analysis of the task stream. The third constraint ensures that the application can no longer view any effects on a store, checked through a split reference counting scheme in the implementation of Diffuse's IR. The split reference counting scheme separates references held by the application from references held by Diffuse's runtime. Temporary stores are demoted from a distributed allocation into a task-local allocation, as described in Section 6.

## <span id="page-7-1"></span>5.2 Memoization of Analyses

The final component of our distributed task fusion pipeline is memoization analysis and code generation (Section 6). The key challenge in memoization is allowing for the analyses to be replayed on *isomorphic* task streams rather than identical task streams. Consider the streams of tasks in Figure 7a, where partitions and launch domains are excluded.

Diffuse may reuse the analysis results from the left stream in Figure 7a on the middle stream, as the pattern of stores among tasks is isomorphic. In contrast, the right task stream in Figure 7a has a different pattern of stores across tasks, particularly the use of S7 in T3. We observe that this problem is identical to *alpha-equivalence*, where each store argument is a bound variable. We identify when two task streams are isomorphic within Diffuse through a conversion to and comparison on a canonical, De-Brujin index-like representation. This representation is shown in Figure 7b. A similar technique has previously been used to avoid enumerating instruction sequences equivalent up to register renaming [9].

```
1 T1([(S1,R), (S2,W)]) 1 T1([(S5,R), (S6,W)]) 1 T1([(S5,R), (S6,W)])
2 T2([(S2,R), (S1,W)]) 2 T2([(S6,R), (S5,W)]) 2 T2([(S6,R), (S5,W)])
3 T3([(S1,R), (S3,W)]) 3 T3([(S5,R), (S7,W)]) 3 T3([(S5,R), (S7,W)])
4 T4([(S3,R), (S1,W)]) 4 T4([(S7,R), (S5,W)]) 4 T4([(S7,R), (S5,W)])
```

(a) Two isomorphic task streams and one differing task stream.

```
1 T1([(0,R), (1,W)])
```

(b) Canonical representations of isomorphic and differing streams.

**Figure 7.** Example of task stream memoization.

#### <span id="page-7-0"></span>6 Kernel Fusion

The final component of Diffuse is a compilation stack to optimize fused tasks. A high-level program representation is required to both perform optimizations like loop fusion and to lower to different backends like GPUs and multi-threaded CPUs. We leverage the MLIR compiler stack, which is extensible and is pre-packaged with many common compiler analyses. We first provide background on MLIR, and then describe the code generation process and optimizations performed within Diffuse. We then discuss how Diffuse's architecture enables the separation of reasoning about distributed programs from the optimization of nested loops.

## 6.1 MLIR Background

We leverage MLIR [38] to build a JIT compiler for Diffuse. MLIR is an extension of LLVM [37] that provides compiler infrastructure for program analyses on higher-level languages than three-address code. The most relevant component of this infrastructure to our work is the notion of a *dialect*, which is an intermediate representation that has user-defined semantics. A key aspect of dialects in MLIR is that a single MLIR program can contain types and operations from multiple dialects, enabling the composition of dialects with different semantics. Compilers built using the MLIR framework run passes over programs that either optimize the operations within a single dialect, or convert between dialects to perform progressive lowering. Diffuse's compiler leverages community-developed dialects and passes to optimize and lower task bodies into CPU and GPU code.

#### 6.2 Generator Functions

To describe Diffuse's compiler, we walk through the stages that a fused task traverses. cuPyNumeric and Legate Sparse developers implement tasks by defining variants that target CPUs or GPUs. To use Diffuse, developers register a *generator* function with Diffuse that returns an MLIR fragment describing the task's computation. We found the integration effort of adding these generator functions to be modest, requiring 50–100 lines of C++ code per operation. We emphasize that only library developers, not end users, must develop MLIR kernels for tasks. Additionally, the integration

```
1 func . func @kernel (
2 % a : memref <? xf64 > ,
3 % b : memref <? xf64 > ,
4 % c : memref <? xf64 >) {
5 % dim = memref . dim %c , 0
6 affine . for % i = 0 to % dim {
7 %0 = affine . load % a [% i ]
8 %1 = affine . load % b [% i ]
9 %2 = arith . addf %0 , %1
10 affine . store %2 , % c [% i ] }}
(a) MLIR generated for an
element-wise addition.
                                        1 func . func @fused_kernel (
                                        2 % a : memref <? xf64 > ,
                                        3 % b : memref <? xf64 > ,
                                        4 % c : memref <? xf64 > ,
                                        5 % d : memref <? xf64 > ,
                                        6 % e : memref <? xf64 >) {
                                        7 % dim = memref . dim %e , 0
                                        8 affine . for % i = 0 to % dim {
                                        9 %0 = affine . load % a [% i ]
                                       10 %1 = affine . load % b [% i ]
                                       11 %2 = arith . addf %0 , %1
                                       12 affine . store %2 , % c [% i ] }
                                       13 affine .for % i = 0 to % dim {
                                       14 %0 = affine . load % c [% i ]
                                       15 %1 = affine . load % d [% i ]
                                       16 %2 = arith . addf %0 , %1
                                       17 affine . store %2 , % e [% i ] }}
                                          (b) Initial body of fused task.
                                                                               1 func . func @fused_kernel (
                                                                               2 % a : memref <? xf64 > ,
                                                                               3 % b : memref <? xf64 > ,
                                                                               4 % d : memref <? xf64 > ,
                                                                               5 % e : memref <? xf64 >) {
                                                                               6 % dim = memref . dim %e , 0
                                                                               7 % c = memref . alloc % dim
                                                                               8 affine .for % i = 0 to % dim {
                                                                               9 %0 = affine . load % a [% i ]
                                                                              10 %1 = affine . load % b [% i ]
                                                                              11 %2 = arith . addf %0 , %1
                                                                              12 affine . store %2 , % c [% i ] }
                                                                              13 affine .for % i = 0 to % dim {
                                                                              14 %0 = affine . load % c [% i ]
                                                                              15 %1 = affine . load % d [% i ]
                                                                              16 %2 = arith . addf %0 , %1
                                                                              17 affine . store %2 , % e [% i ] }}
                                                                               (c) After temporary elimination.
                                                                                                                      1 func . func @fused_kernel (
                                                                                                                      2 % a : memref <? xf64 > ,
                                                                                                                      3 % b : memref <? xf64 > ,
                                                                                                                      4 % d : memref <? xf64 > ,
                                                                                                                      5 % e : memref <? xf64 >) {
                                                                                                                      6 % dim = memref . dim %e , 0
                                                                                                                      7 affine . par % i = 0 to % dim {
                                                                                                                      8 %0 = affine . load % a [% i ]
                                                                                                                      9 %1 = affine . load % b [% i ]
                                                                                                                     10 %2 = arith . addf %0 , %1
                                                                                                                     11 %3 = affine . load % d [% i ]
                                                                                                                     12 %4 = arith . addf %2 , %3
                                                                                                                     13 affine . store %2 , % e [% i ] }}
                                                                                                                        (d) Fully optimized fused task.
```

Figure 8. Fused MLIR kernel for three way element-wise addition traversing the compilation pipeline. The initial kernel is created by sequentially composing two of the generated task bodies in Figure [8a.](#page-8-0)

effort was incremental—as more tasks were implemented with MLIR generators, Diffuse could exploit more fusion. An example generated MLIR fragment by cuPyNumeric for an element-wise addition operation is shown in Figure [8a.](#page-8-0)

The generated MLIR fragment in Figure [8a](#page-8-0) contains multiple dialects: 1) stores are mapped onto the memref dialect, which provides stronger aliasing guarantees than raw pointers; 2) dense iteration is mapped onto the affine dialect, a target for polyhedral compilation [\[18\]](#page-13-12); and 3) the computation itself is mapped onto the arith dialect, containing arithmetic operations. Using MLIR, other dialects can be used to express higher level operations, like dense and sparse tensor algebra with the linalg and sparse\_tensor dialects.

