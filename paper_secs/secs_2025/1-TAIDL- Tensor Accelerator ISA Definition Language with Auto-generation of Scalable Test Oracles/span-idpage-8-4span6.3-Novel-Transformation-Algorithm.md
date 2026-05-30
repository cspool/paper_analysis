# <span id="page-8-4"></span>6.3 Novel Transformation Algorithm

```
1 def transform( instrs ) :
2 state = init_cregs () ○A
3 hlo_txt = prologue () ○B
4 for instr in instrs :
5 attr = instr . attr # calling attributes
6 compute = instr . compute # Tensor compute
7 compute = compute . resolve ( attr , state ) ○C
8 stmts = expand_blocks ( compute ) ○D
9 for stmt in stmts :
10 if stmt . op == TENSOR_BUFFER_READ : ○E (_)
11 hlo_txt += gen_slice ( stmt )
12 elif stmt . op == TENSOR_BUFFER_WRITE : ○F (_)
13 hlo_txt += gen_dy_up_slice ( stmt )
14 elif stmt . op == XLA_HLO_TENSOR_OP : ○G (ℎ_)
15 hlo_txt += stmt
16 elif stmt . op == ASSIGN_STMT : ○H ()
17 state = state . update ( stmt )
18 elif stmt . op == ASSERT_STMT : ○I ()
19 assert ( stmt )
20 hlo_txt += epilogue ()
21 return hlo_txt
22
23 def expand_blocks( compute ) :
24 stmts = []
25 for block in compute :
26 if block . op == REPEAT_BLOCK : ○J
27 for i in range ( block . iter ) :
28 stmts += expand_blocks ( block . body ( i ) )
29 elif block . op == IF_BLOCK : ○K
30 sel = select ( block , block . condition )
31 stmts += expand_blocks ( sel )
32 else :
33 stmts += block
34 return stmts
```

Figure 16: Transforming an ISA-specific kernel function into a tensor computation graph in XLA-HLO IR.

Figure [16](#page-8-3) shows the pseudocode for transforming a kernel function into an XLA-HLO IR representing a semantically equivalent tensor computation. The algorithm (transform) transforms a stream of instructions (instrs) into a tensor computation graph (hlo\_txt).

It first initializes the control registers (state) (A) and generates the prologue of the tensor computation graph (B). The prologue consists of the initialization of the tensor buffers and the HBM. It then processes the stream of instructions in sequential order.

Since the expressions in *compute* are only the functions of calling attributes and control registers, they can be resolved by constant propagation ( $\mathbb{C}$ ). This is followed by recursively expanding the blocks in the *compute* into a list of statements ( $\mathbb{D}$ ). The REPEAT blocks are unrolled ( $\mathbb{D}$ ) while the IF blocks select the appropriate branch ( $\mathbb{C}$ ). It then translates the list of statements (stmts) into XLA-HLO syntax. The tensor buffer reads and writes are converted into XLA-HLO slice<sup>9</sup> and dynamic\_update\_slice<sup>10</sup> operators ( $\mathbb{E}$  and  $\mathbb{E}$ ). The XLA-HLO tensor operations are appended to the tensor computation graph as is ( $\mathbb{C}$ ). The assignment statements update the control registers (state) ( $\mathbb{E}$ ). The assertion statements are evaluated ( $\mathbb{D}$ ). It finally returns the tensor computation as a XLA-HLO graph, which then can be compiled by the XLA compiler.

In summary, *assign* and *assert* statements on control registers and calling attributes are statically analyzed via constant folding, while *tb\_read* and *tb\_write* on tensor buffers are replaced by slice and dynamic\_update\_slice operators with no changes to *hlo\_op*.

```
1 # TAIDL: %In:65536xi8
```

Figure 17: Snippet of XLA-HLO IR emitted by the transformation algorithm (Figure 16) for TPUv1 instruction call read\_weights(addr=0) when control register set push is 2.

Figure 17 shows a snippet of the XLA-HLO graph emitted by TAIDL-TO for TPUv1 instruction read\_weights. Instruction semantics in Figure 6 are transformed into XLA-HLO operators one-byone. Control registers (push) and attributes (addr) are statically analyzed and known when emitting XLA-HLO IR for an instruction.

#### 6.4 Programmer's View: Compiling the Oracle

matmul\_amx.compile() is called to compile the kernel function matmul\_amx into an executable. First, the kernel function is transformed into a XLA-HLO graph using the transformation algorithm discussed in §6.3. The XLA-HLO graph is then compiled into an executable using the XLA compiler present in jaxlib library. XLA generates a serialized executable in protobuf format (.pb). XLA supports GPU as a backend platform, allowing for GPU-accelerated simulations using the generated test oracle library.

#### 6.5 Programmer's View: Running the Oracle

A compiled kernel function can be directly invoked as a callable Python function, allowing for easy integration within an ML model. For example, C = matmul\_amx.run(A, B) loads the compiled executable with the input tensors A and B and stores the result in C.

The input tensors are NumPy arrays that are passed as arguments to the kernel function invocation. The simulation is executed on CPU or GPU, based on the backend platform used for compilation.

```
1 def forward(A: np.ndarray, B: np.ndarray):
2    C = matmul_amx.run(A,B)  # Simulate AMX on TAIDL-TO
3    D = np.maximum(0,C)  # Execute host code natively
4    X = nn_gemmini.run(A,B)  # Simulate Gemmini on TAIDL-TO
5    assert((D == X).all())  # Execute host code natively
```

Figure 18: Python function simulating multiple kernel functions (treated as a callable function) integrated with host code, which is executed using the native Python interpreter.

TAIDL-TO supports simulation of multiple kernel functions integrated with host code as shown in Figure 18. Similar to Intel SDE, TAIDL-TO only simulates accelerator instructions and executes the host code natively using the default Python interpreter.

TAIDL-TO also provides debugging capabilities. A programmer can add debug locations within a kernel function (Figure 15 line 17) to log the values stored in scratchpads and control registers. This is similar to nki.language.device\_print [66] provided by AWS NKI.

#### 6.6 Discussion

Scalability. TAIDL's design choice of using XLA-HLO for instruction semantics plays a key role in making TAIDL-TO scalable for large kernels. This allows us to automatically generate TAIDL-TO that uses XLA-HLO IR, a domain-specific IR for tensor computations, and compiles using tensor compilers like XLA. Additionally, XLA automatically generates highly parallelized executables that take advantage of multi-threading as well as GPU acceleration. We evaluate the scalability of TAIDL-TOs generated from TAIDL against existing instruction-level test oracles in §7.

Retargetability. The novel transformation algorithm in Figure 16 is parameterized by TAIDL constructs like the instruction semantics (instr.compute) and data model (prologue), making it retargetable to any accelerator ISA written in TAIDL. This minimizes the effort needed to develop scalable test oracles for new tensor accelerators.

Software Readiness. TAIDL-TO enables early development of accelerator software libraries and compiler backends by providing a functional kernel library that mimics the target ISA. During the presilicon phase, these software components are written and tested using TAIDL-TO. The resulting software is forward-compatible and can be reused on post-silicon chips. For instance, x86 assembly code can be generated from Figure 15 by removing "amx.api.". Thus, TAIDL-TO promotes software readiness by bridging the gap between architecture prototyping and production deployment.

#### <span id="page-9-0"></span>7 Scalability of Auto-generated TAIDL-TOs

We evaluated the scalability of the auto-generated TAIDL-TOs by comparing the simulation time of the generated TAIDL-TOs against the existing instruction-level test oracles – Gemmini Spike and Intel SDE. Gemmini Spike [103] is a RISC-V ISA simulator that models the Gemmini ISA [102]. Intel SDE [39] is a binary translation-based simulator that models the x86 ISA [37]. We selected these two simulators based on the availability of well-documented ISA semantics, open-source correctness testing infrastructure, and the granularity of simulation (instruction-level) and precision (bit-precise).

<sup>9</sup>https://openxla.org/xla/operation\_semantics#slice

 $<sup>^{10}</sup> https://openxla.org/xla/operation\_semantics\#dynamic update slice$ 

#### <span id="page-10-2"></span>(a) DIM = 16(b) DIM = 64(c) DIM = 256Gemmini Spike Simulation time (in ms) TAIDL-TO (GPU) 10 TAIDL-TO (CPU) 5600x 10 620x [1200x] 10 88x 10 10 KB Kernel size Kernel size Kernel size

#### TAIDL-TO for Gemmini ISA vs. Gemmini Spike

Figure 19: The simulation time (lower is better) for the tiled matrix multiplication kernel on TAIDL-TO and Gemmini Spike. The X-axis represents the kernel size, i.e., the total size of input and output tensors. Both axes are log-scaled. TAIDL-TO (GPU) is slower than TAIDL-TO (CPU) for DIM = 16 due to limited parallelization opportunities, but scales better as kernel size grows.

#### 7.1 Experimental Setup

*TAIDL-TO Generation.* We defined TAIDL for Gemmini ISA and Intel AMX/AVX-512 instructions based on the respective ISA manuals [37, 102] and auto-generated TAIDL-TOs as discussed in §6.

Machine Setup. We used a GPU server machine with a 64-core Intel Xeon Platinum 8358 CPU and NVIDIA A100 GPU for all evaluations. In §7.2, we evaluated the performance of TAIDL-TO and Gemmini Spike on the tiled matrix multiplication benchmark. In §7.3, we evaluated the performance of TAIDL-TO and Intel SDE on benchmarks from the Intel oneAPI Deep Neural Network Library (oneDNN) [50]. We compiled two versions of TAIDL-TO, one each with XLA GPU backend (labeled as "TAIDL-TO (GPU)") and XLA CPU backend (labeled as "TAIDL-TO (CPU)"). We observed that simulations using Intel SDE and Gemmini Spike do not utilize GPU.

Metrics. We measured the average simulation time (in milliseconds) across multiple runs. Simulation time of only accelerator instructions is measured. We evaluated scalability by varying the benchmark kernel size, i.e., the total size of input and output tensors.

Simulation correctness. In addition to the scalability analysis of TAIDL-TOs, we also tested whether the output generated is functionally correct. For Gemmini benchmarks, we tested the TAIDL-TO outputs against Gemmini's RTL simulation (or Gemmini Spike if RTL simulation takes more than an hour). For Intel oneDNN benchmarks, we tested the output of TAIDL-TOs against native execution on a Sapphire Rapids machine (Intel Xeon Gold 5415+ CPU). The outputs matched exactly to that of baselines, i.e., are bit-accurate.

