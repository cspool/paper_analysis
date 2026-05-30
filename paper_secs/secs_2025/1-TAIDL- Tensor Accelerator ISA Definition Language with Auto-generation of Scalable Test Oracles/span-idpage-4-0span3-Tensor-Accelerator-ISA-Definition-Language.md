# <span id="page-4-0"></span>3 Tensor Accelerator ISA Definition Language

Tensor Accelerator ISA Definition Language (TAIDL) is a domainspecific language (DSL) designed to describe the Instruction Set Architecture (ISA) of a tensor accelerator. An ISA provides information about the user-programmable storage units (collectively termed as *data model*) like scratchpads, and the instructions that perform computations like data movement and compute on the storage units. TAIDL provides a high-level understanding of the computational capabilities of the accelerator without going into its implementation details (microarchitecture design).

A TAIDL definition has two main components – the data model definition and the instruction semantics. Next, we discuss the syntax and terminology of TAIDL with the help of a simple example of TPUv1 [71] and its instruction read\_weights. In §4, we provide more case studies showing the expressive power of TAIDL.

#### 3.1 Data Model Definition

The data model in TAIDL is designed to be flexible enough to cover a variety of storage units present in different tensor accelerators. It consists of two types of storage units - tensor buffers and control registers. Figure 3 shows the core syntax of the data model definition in TAIDL. Figure 4 shows the data model for TPUv1.

```
data_model ::= tbuffers cregs
tbuffers ::= { [ ld ] ( tshape ) ( element_type ) }
tshape ::=
```

Figure 3: Core Syntax of Data Model in EBNF [109]. The terminals are colored in brown.

```
1 # Data Model: Tensor Buffers
2 [unified_buffer] (96K) (256xi8);
3 [accumulator]
                    (4K)
                          (256xi32)
4 [weights]
                    ()
                          (256x256xi8):
5 [fifo]
                    (4)
                          (256x256xi8);
6 # Data Model: Control Registers
           = 0;
7 occupancy
8 push
9 pop
```

Figure 4: TAIDL definition of TPUv1 [71] data model.

*Tensor Buffers*. The tensor buffers represent storage units that store the input, output, and intermediate tensor data of an accelerator. They are defined as multi-dimensional arrays of base elements. A base element itself can be defined as a multi-dimensional data type.

Lines 2 to 5 of Figure 4 define the storage buffers of TPUv1. The Unified Buffer in TPUv1 has 96K rows of 256-length vectors of i8. Since the granularity of data access to the Unified Buffer is a row, we model it as the base element. Thus, it is represented as a one-dimensional buffer of size (96K) with base elements of (256xi8) (line 2). The systolic array in TPUv1 performs weight-stationary computation with a pre-loaded weight matrix (line 4). The FIFO buffer is used to store weights before the systolic array loads them. It holds a maximum of four 256×256 matrices of i8 (line 5).

Multi-dimensionality of tensor buffers and base elements plays an important role in supporting various shapes and sizes of storage units often observed in tensor accelerators (examples in §4.1-§4.2).

 $<sup>^{1}</sup> https://openxla.org/xla/operation\_semantics\#element-wise\_unary\_functions$ 

 $<sup>^2</sup> https://openxla.org/xla/operation\_semantics\#reduce precision$ 

Control registers. The control registers represent values in control units that are exposed to programmer control. These values represent the state of the accelerator and control the execution of the instructions. Semaphore registers (set/unset at asynchronous calls), configuration flags (if user-controlled, like dataflow reconfigurable), and counters are some use cases of control registers.

Lines 7 to 9 of Figure 4 show the registers that control the state of the FIFO buffer in TPUv1. occupancy stores the number of layers occupied in the FIFO buffer. push and pop store the indices of the layers that are being pushed and popped from the FIFO buffer.

We discuss the reasoning behind separating the definition of control registers and tensor buffers in §4.3.

Global Memory (HBM). Most accelerators have access to global memory that can be accessed by the instructions. We refer to this global memory as HBM (High Bandwidth Memory) hereafter. HBM need not be explicitly defined in TAIDL. HBM is represented as a one-dimensional buffer with base elements of (i8).

#### 3.2 Instruction Semantics

The instruction semantics in TAIDL specifies the behavior of each instruction in the accelerator ISA, without going into the implementation details of the accelerator. Figure 5 shows the core syntax of instruction semantics in TAIDL. Figure 6 shows the semantics of a TPUv1 instruction to load weights into the FIFO buffer.

```
isa ::= { instruction }\ninstruction ::= [Id] ( attributes ) compute
attributes ::= Id { , Id } | \( \epsilon \)
compute ::= block compute | stmt compute | \( \epsilon \)
block ::= repeat_block | if_block
repeat_block ::= REPEAT ( lvar , aexp ) { compute }\nif_block ::= IF ( bexp ) { compute } ELSE { compute }
stmt ::= tb_read | tb_write | hlo_op | assign | assert
```

Figure 5: Core Syntax of Instruction Semantics in EBNF [109]. The terminals are colored in brown. aexp and bexp refer to an arithmetic expression and a boolean expression, respectively. lvar refers to a local variable name.

```
1 [read_weights] (addr)
2 assert(occupancy < 4);
3 %In:65536xi8
```

Figure 6: TAIDL definition of a TPUv1 [71] instruction.

Calling Attributes. Each instruction in the accelerator ISA can take inputs as operands, similar to the register numbers in a RISC-V instruction. We refer to these inputs as calling attributes, and any stream of instructions written in the ISA contains these attributes. For example, the calling attribute to read\_weights is the HBM address (addr) from which the weights are to be read (line 1).

Tensor Computation. The compute of an instruction is defined as a tensor computation on data stored in the tensor buffers and HBM. They can refer to the calling attributes, tensor buffers, and control registers. In addition to operational semantics, an instruction also needs to satisfy certain constraints to avoid undefined behavior.

In TAIDL, we model these as five types of statements:

- Tensor Read: tb\_read statement reads a slice from a tensor buffer and writes to a tensor intermediate (line 3). TAIDL supports Python-like array slicing syntax.
- Tensor Write: tb\_write statement updates a slice of a tensor buffer with values from a tensor intermediate (line 5).
- Tensor Operation: hlo\_op statement performs a tensor operation on a tensor intermediate and writes to another tensor intermediate (line 4). TAIDL supports tensor operations present in XLA-HLO IR. We discuss this choice with examples in §4.4.
- Control Register Assignment: *assign* statement updates the value of a control register. The assignment value is an expression over calling attributes and control registers (lines 6 and 7).
- Assertion: assert statement specifies the constraints that must be satisfied for an instruction to be valid. It is a Boolean expression over calling attributes and control registers (line 2).

The *compute* is further augmented with IF and REPEAT blocks to support instructions with dynamic shapes and control flow. We discuss the role of this augmentation in §4.5.

#### <span id="page-5-0"></span>4 Expressivity of TAIDL

We demonstrate the expressive power of TAIDL by showing how each design choice covers various nuances observed in existing tensor accelerator designs. We present interesting snippets of TAIDL definitions with the complete definitions available in the artifact.

