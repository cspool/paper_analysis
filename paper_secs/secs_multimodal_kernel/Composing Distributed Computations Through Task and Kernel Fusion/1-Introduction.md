# 1 Introduction

A modern trend in distributed programming is to develop drop-in implementations of popular sequential libraries like NumPy or SciPy that automatically scale to distributed machines while maintaining the semantics of the original library [\[12,](#page-13-0) [13,](#page-13-1) [30,](#page-13-2) [60\]](#page-15-0). To achieve distribution, these drop-in replacement libraries are implemented by translation to a distributed task-based runtime system [\[7,](#page-13-3) [11,](#page-13-4) [15,](#page-13-5) [26,](#page-13-6) [41\]](#page-14-0). Libraries map computations to a stream of tasks issued to the runtime, and map data on to runtime-managed distributed collections. Tasks are user-defined functions, whose bodies we call kernels, that operate on subsets of the distributed collections. The runtime is responsible for extracting parallelism from the input sequence of tasks and for computing the synchronization and communication required between tasks. This architecture enables distributed libraries to be built independently and then composed freely, as the runtime system is responsible for scheduling parallel work and maintaining coherence of distributed data.

However, the same abstractions that yield important composition properties internally and externally to these distributed libraries can result in degraded end-to-end performance. The task decomposition of library operations results in tasks that may be optimized individually but can have poor data locality and allocate much more temporary data than a different program organization that breaks down the abstraction boundaries by fusing tasks together both within the operations of a particular library and across library boundaries. As the task-based runtime system is issued a stream of tasks after library abstraction boundaries have been traversed, the runtime has the opportunity to fuse the tasks

from different libraries together, which in turn enables the fusion of the kernels of nested loops within fused tasks. Fusion by the task-based runtime allows for these optimizations to be performed without being limited to the semantics of any particular task-based library.

Prior works such as Weld [\[44\]](#page-14-1) and Split Annotations [\[45\]](#page-14-2) have developed techniques to perform fusion across library boundaries, but only for shared memory libraries. Distributed memory complicates program analyses, as distributed data requires communication when shared data is written to and read from by different nodes. For example, a sequence of element-wise operations on a pair of distributed arrays may or may not be fusible depending on whether the arrays are aliases of the same distributed data. We do not consider the problem of automatic parallelization [\[2,](#page-12-0) [5,](#page-12-1) [19,](#page-13-7) [39,](#page-14-3) [57\]](#page-15-1); the task-based programs we consider are already (implicitly) parallel. We focus on the efficient composition of independentlywritten parallel, distributed programs.

We present Diffuse, a system that dynamically performs task and kernel fusion for distributed, task-based runtime systems, transparently achieving optimizations found in handtuned programs. Diffuse reasons over a task-based IR of distributed computation, modeling computation as a sequence of tasks operating on partitioned data (Section [3\)](#page-2-0). Diffuse's IR is scale-free, meaning that the size of the IR and analyses on it are independent of the size of the target machine. Diffuse uses this IR to perform a dynamic dependence analysis to fuse tasks in a distributed-memory setting (Section [4\)](#page-4-0). Diffuse pairs task fusion with a JIT compiler based on MLIR [\[38\]](#page-14-4) that fuses and optimizes kernels within dynamically fused tasks, enabling data reuse across independent tasks (Section [6\)](#page-7-0). By analyzing a task-based IR, Diffuse's optimizations are not tied to the semantics of any particular library.

We implement Diffuse as a middle layer between highlevel task-based libraries and the low-level Legion runtime system [\[15\]](#page-13-5). To demonstrate Diffuse, we modify the implementations of the distributed libraries cuPyNumeric [\[12\]](#page-13-0) and Legate Sparse [\[60\]](#page-15-0) to target Diffuse's IR, and to expose their task implementations in MLIR for Diffuse's compiler to process. Diffuse then performs dynamic analyses to fuse the tasks and kernels issued by these libraries before forwarding the optimized tasks to Legion. As a result, programmers using cuPyNumeric and Legate Sparse benefit from Diffuse without modifying their applications.

To evaluate Diffuse, we apply it to micro-benchmarks and several full scientific computing applications developed in cuPyNumeric and Legate Sparse, including sparse Krylov solvers and physical simulations. We compare against the standard implementations of cuPyNumeric and Legate Sparse and show that Diffuse achieves 1.86x speedup on average (geo-mean) over unmodified applications on up to 128 GPUs. We additionally compare against the high-performance MPIbased PETSc [\[8\]](#page-13-8) library and show that Diffuse enables naturally written NumPy and SciPy Sparse programs to match or

```
1 import cunumeric as np
 2 grid = np . random . rand (( N +2 , N +2) )
 3 # Create multiple aliasing views
 4 # of the distributed grid array .
 5 center = grid [1: -1 , 1: -1]
 6 north = grid [0: -2 , 1: -1]
 7 east = grid [1: -1 , 2: ]
 8 west = grid [1: -1 , 0: -2]
 9 south = grid [2: , 1: -1]
10 for i in range ( niters ) :
11 avg = center + north + \
12 east + west + south
13 work = 0.2 * avg
                                                         (1,0) (1,1) (1,2) (1,3)
                                                         (0,0) (0,1) (0,2) (0,3)
                                                         (3,0) (3,1) (3,2) (3,3)
                                                         (2,0) (2,1) (2,2) (2,3)
                                                             (1,1) (1,2)
                                                             (0,1) (0,2)
                                              (1,0) (1,1)
                                              (2,0) (2,1)
                                                west
                                                              south
                                                                              east
                                                               north
                                                             (3,1) (3,2)
                                                             (2,1) (2,2)
                                                                           (1,2) (1,3)
                                                     grid center
```

(a) cuPyNumeric source code.

<span id="page-1-4"></span><span id="page-1-3"></span><span id="page-1-2"></span><span id="page-1-1"></span>14 center [:] = work

```
ted lines denote communication.
1 # ADD , MULT and COPY are in
2 # cuNumeric 's implementation .
3 ALLOC ARRAY t1
4 ADD ( center , north , t1 )
5 ALLOC ARRAY t2
6 ADD ( t1 , east , t2 )
7 ALLOC ARRAY t3
8 ADD ( t2 , west , t3 )
9 ALLOC ARRAY avg
10 ADD ( t3 , south , avg )
11 ALLOC ARRAY work
12 MULT (0.2 , avg , work )
13 COPY ( work , center )
                                        1 # FUSED_ADD_MULT is a new
                                        2 # task generated by Diffuse .
                                        3 ALLOC ARRAY work
                                        4 FUSED_ADD_MULT (
                                        5 center ,
                                        6 north ,
                                        7 east ,
                                        8 west ,
                                        9 south ,
                                       10 0.2 ,
                                       11 work
                                       12 )
                                       13 COPY ( work , center )
```

(c) Stream of tasks and allocations issued by the main loop.

```
1 # ADD , MULT and COPY are
2 # elementwise operators .
3 def ADD (a , b , c ) :
4 for i , j in a :
5 c [i , j ] = a [i , j ] + b [i , j ]
6 def MULT (s , a , b ) :
7 for i , j in a :
8 b [i , j ] = s * a [i , j ]
9 def COPY (a , b ) :
10 for i , j in a :
11 b [i , j ] = a [i , j ]
```

(e) Tasks invoked during standard execution.

(d) Operation stream after Diffuse's optimization.

(b) 4-node execution, colors denote cells held by each node. Dot-

```
1 # FUSED_ADD_MULT performs
2 # the scaled five - way add .
3 def FUSED_ADD_MULT (
4 a , b , c , d , e , s , out
5 ) :
6 for i , j in a :
7 out [i , j ] = s * (
8 a [i , j ] + b [i , j ]
9 + c [i , j ] + d [i , j ]
10 + e [i , j ])
```

(f) Fused task generated by Diffuse.

Figure 1. Execution example of Diffuse on a distributed, multi-GPU cuPyNumeric 5-point stencil application.

exceed the performance of PETSc (1.4x geo-mean speedup). Finally, we show that Diffuse is able to find fusion and optimization opportunities missed by the original application developers, achieving 1.23x speedup on average (geo-mean) over already hand-optimized code.

