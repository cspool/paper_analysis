# <span id="page-12-3"></span>8.2 Case Study: Simulating End-to-End Model

In  $\S$ 7, we evaluated the scalability of TAIDL-TO against existing instruction-level functional simulators over small-to-medium kernels. Next, we simulate an end-to-end I-BERT [75] transformer model on TAIDL-TOs generated for Gemmini ISA (DIM = 256). We select I-BERT since Gemmini, by default, supports only integer operations with additional support for I-BERT's quantized activations.

We measured the functional simulation time of I-BERT (12 encoder layers, 768 embedding size) with sequence length of 512 using the same machine setup as §7. TAIDL-TO completed the simulation in just 2.4 sec (GPU disabled) and 0.8 sec (GPU enabled), in contrast to over 50 mins required by Gemmini Spike. This shows that TAIDL-TO enables fast and practical end-to-end model simulation.

#### 9 Related Works

Instruction Specification Languages. Sail [8, 52] provides a language to describe the instruction semantics of IBM Power, ARMv8, RISC-V, and MIPS. Compared to TAIDL, which targets tensor accelerators, Sail supports only scalar instructions. Hydride [77] and VeGen [22] support vector instruction semantics derived from vendor-provided pseudocode using bitvector representations. Instruction-Level Abstraction (ILA) [56] defines a formal model of hardware execution of instructions and is designed for verification of hardware behavior.

DSL Specification Languages. ODS [30] and TableGen [29] are used in MLIR to define the syntactic components like operand types and constraints of new operations, but not their meaning. For example, the AIEVecOps.td [58] doesn't model semantics but only the syntax of AIE vector instructions along with natural language descriptions.

Performance Models. Prior works on performance modeling, like Timeloop [99] and MAESTRO [79] model memory hierarchies and dataflow mappings, but not ISA semantics. These are suited for architectural exploration, whereas TAIDL models functional behavior of instructions, enabling correctness testing of the software stack.

Hardware Simulators. Hardware architects measure performance metrics using simulator platforms, including discrete-event simulators like gem5 [15] and Structural Simulation Toolkit (SST) [105], event-driven cycle-level timing simulators like ZSim [108], and RTL simulators like Synopsys VCS [67] and Verilator [111]. These tools simulate details of microarchitectural elements, while our work focuses on instruction-level functional simulation of ISA.

Accelerator Design Languages. Existing accelerator design languages (ADLs) [19, 76, 78, 81, 96, 97, 117, 120, 121] are used to design microarchitectures and generate the HLS or RTL code for the accelerators. These languages do not define the ISA but design optimized computational units for a particular application/kernel. Unlike ADLs, TAIDL's primary use case is to aid hardware architects in defining an ISA and its semantics for their tensor accelerators.

Accelerator Programming Languages. TensorFlow [2] and JAX [17] allow users to write high-level programs and target TPUs using the XLA-TPU compiler, but do not expose interfaces for adding new instruction definitions. Exo [57] supports the definition of custom hardware instructions, but lacks memory addressing in these definitions and requires special allocation/deallocation instructions, making it inadequate for defining tensor accelerator ISAs.

#### 10 Conclusion

We present TAIDL, an ISA specification language for tensor accelerators that captures the diverse memory hierarchies and compute capabilities of modern accelerator designs. TAIDL allows accelerator designers to express the intent of the instructions without delving into hardware implementation details, providing a *standardized*, concise, human-readable specification of ISA semantics. This standardization enables us to automatically generate fast and scalable test oracle platforms for software development.

TAIDL enables the systems and compilers community to develop and test software more efficiently, fostering adoption of emerging tensor accelerators in ML ecosystems and broader collaboration between hardware and software research communities.

 $<sup>^{11}</sup> https://github.com/exo-lang/exo/issues/803$ 

We have open-sourced the TAIDL library to the community to facilitate the development of tensor accelerator ISAs. The repository [\(https://github.com/act-compiler/taidl\)](https://github.com/act-compiler/taidl) includes detailed language documentation, example ISA definitions, and tooling to generate test oracle platforms; we welcome contributions and feedback.

