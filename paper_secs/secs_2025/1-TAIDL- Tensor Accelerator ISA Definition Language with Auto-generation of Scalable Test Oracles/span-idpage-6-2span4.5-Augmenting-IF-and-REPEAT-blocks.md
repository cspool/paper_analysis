# <span id="page-6-2"></span>4.5 Augmenting IF and REPEAT blocks

Recall from [§3](#page-4-0) that TAIDL models programmer-exposed accelerator state like configuration flags as control registers and instruction side-effects as assignments to these control registers (assign).

While XLA-HLO can represent control flow over tensor data stored in tensor buffers, it is not sufficient to represent control flow parameterized by the calling attributes and the control registers. We solve this by augmenting compute with IF and REPEAT blocks.

```
1 [ config_execute_dataflow ] (new_dataflow_value)
2 dataflow_flag = new_dataflow_value
3
4 [ matmul_compute_preloaded ] (rs1, rs2)
5 IF (dataflow_flag == 0) {
6 ... # tensor computation for output - stationary
7 } ELSE {
8 ... # tensor computation for weight - stationary
9 }
```

Figure 10: TAIDL definition of Gemmini [\[51\]](#page-15-1) instructions that model switching of dataflow via a control register.

IF block. An IF block represents conditional branching, allowing for different tensor operations to be executed based on the accelerator state. The argument to an IF block is a boolean expression over calling attributes and control registers (Figure [5\)](#page-5-3). This is useful for several instructions in Gemmini ISA [\[102\]](#page-17-5) where the instruction definition is overloaded for multiple dataflow configurations. Figure [10](#page-6-4) shows a snippet of the TAIDL definition of a Gemmini instruction that performs a different operation based on the dataflow configuration, which is modeled as a control register dataflow\_flag.

```
1 [ tdpbusd ] (dst, src0, src1)
2 REPEAT (m , 16) {
3 REPEAT (k , 16) {
4 REPEAT (n , 16) {
5 ... # perform vector dot - product on 4 bytes ○1
6 }
7 }
8 }
```

Figure 11: Alternate TAIDL definition of AMX instruction tdpbusd (Figure [2\)](#page-2-0) using nested REPEAT blocks. The inner loop body ○1 is a vector-vector dot-product on 4 bytes of input tiles. We skip the inner loop body definition for brevity.

REPEAT block. A REPEAT block is used to repeat a tensor operation. Syntactically, the argument to the REPEAT block is an arithmetic expression over calling attributes and control registers (Figure [5\)](#page-5-3). Figure [11](#page-6-5) shows an example of a TAIDL definition with nested RE-PEAT blocks. Given XLA-HLO's rich operator set, most instances of REPEAT blocks have equivalent semantics without REPEAT blocks. For example, TAIDL definition of AMX instruction tdpbusd in Figure [11](#page-6-5) is semantically equivalent to the compact definition without REPEAT blocks in Figure [2](#page-2-0) (b). We have not observed a case where an instruction semantics necessitates the usage of REPEAT blocks.

<sup>3</sup>[https://openxla.org/xla/operation\\_semantics#bitcastconverttype](https://openxla.org/xla/operation_semantics#bitcastconverttype)

