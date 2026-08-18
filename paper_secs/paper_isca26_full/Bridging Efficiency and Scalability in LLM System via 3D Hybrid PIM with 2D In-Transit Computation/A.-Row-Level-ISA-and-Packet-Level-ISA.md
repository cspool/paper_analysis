# *A. Row-Level ISA and Packet-Level ISA*

To program SRAM-PIM and NoC at row level in SIMD, the defined instructions are shown in Table I. The SRAM\_Write and SRAM\_Comp instructions are used to write the weights for configuration and to write the input vector to the SRAM for computing. For SRAM\_Write, the source address (SRC) and length (Len) are bank-relative DRAM addresses; data are broadcast from the same SRC and Len to all SRAM-PIM macros in the bank (heterogeneous macros receive the same logical segment). SRAM\_Comp loads input from SRC and writes matrix multiply results to DST with Len elements. RoPE is mapped explicitly via NoC\_Exchange for data rearrangement followed by DRAM-PIM EWMUL (Section IV-C). CompAir's addressing is confined to DRAM banks, while SRAM-PIM operations (weight reloading and computing) are instruction-granular with fixed dataflow, eliminating SRAM addressing overhead.

TABLE I ROW-LEVEL ISA FOR NOC AND SRAM-PIM

| INST         | OP          | SRC  | DST  | NUM1   | NUM2    |
|--------------|-------------|------|------|--------|---------|
| NoC Scalar   | +=,-=,*=,/= | Addr | Addr | Mask   | Config  |
| NoC Access   | Rd, Wr      | Addr | Addr | Mask   | Const   |
| NoC BCast    | /           | Addr | Addr | Mask   | SrcBank |
| NoC Reduce   | +=,-=,*=,/= | Addr | Addr | Mask   | DstBank |
| NoC Exchange | T+/-,R+/-   | Addr | Addr | Offset | Group   |
| SRAM Write   | /           | Addr | /    | Len    | /       |
| SRAM Comp    | /           | Addr | Addr | Len    | /       |

Within each bank, NoC-related instructions are operated at scalar granularity. From the programming perspective, we view the NoC purely as a computational component in this ISA level, without considering the communication behavior within the NoC. Five NoC-related instructions are designed.

NoC\_Scalar is responsible for once computation in router and NoC\_Access is used to read/write the Curry ALU's registers: the 64-bit Mask is used to indicate whether 64 routers of a channel accept the computation task.

NoC\_Reduce and NoC\_BCast perform DRAM banklevel reduction and broadcasting, using Mask to determine SRAM-PIM macro participation. Both support 4 parallel trees, with DstBank/SrcBank specifying the target/source bank.

NoC\_Exchange differs by allowing both intra-row and inter-bank data exchange, where T and R denote inter-bank and intra-row swaps, +/- indicate inversion, and Offset and Group define swap targets as (x+Offset) % Group. For RoPE, NoC\_Exchange(R-,SrcRow,DstRow,1,2) can be used to express the exchange.

TABLE II PACKET-LEVEL ISA FOR NOC

| Type   | Data | IterNum | Path[0]    | Path[1]      | Path[2] | Path[3]     |
|--------|------|---------|------------|--------------|---------|-------------|
| 4b     | 16b  | 4b      | 12b        | 12b          | 12b     | 12b         |
| Path.X |      | Path.Y  | Path.WrReg | Path.IterTag |         | Path.Opcode |
| 4b     |      | 4b      | 1b         | 1b           |         | 2b          |

Table II shows the packet information at the time of router execution. Where Type is used to indicate the instruction information, currently includes seven types: None, Scalar, Reduce, Exchange, Broadcast, Read, and Write. The Data field contains BF16-formatted payload within the packet. IterNum specifies the iteration count for the computational path, while Path defines the router sequence for each computation step. The control signals include: WrReg for register write-enable in CurryALU, IterTag which triggers dynamic ArgReg updates via IterArg and IterOp after computation.

# *A. Row-Level ISA and Packet-Level ISA*

To program SRAM-PIM and NoC at row level in SIMD, the defined instructions are shown in Table I. The SRAM\_Write and SRAM\_Comp instructions are used to write the weights for configuration and to write the input vector to the SRAM for computing. For SRAM\_Write, the source address (SRC) and length (Len) are bank-relative DRAM addresses; data are broadcast from the same SRC and Len to all SRAM-PIM macros in the bank (heterogeneous macros receive the same logical segment). SRAM\_Comp loads input from SRC and writes matrix multiply results to DST with Len elements. RoPE is mapped explicitly via NoC\_Exchange for data rearrangement followed by DRAM-PIM EWMUL (Section IV-C). CompAir's addressing is confined to DRAM banks, while SRAM-PIM operations (weight reloading and computing) are instruction-granular with fixed dataflow, eliminating SRAM addressing overhead.

TABLE I ROW-LEVEL ISA FOR NOC AND SRAM-PIM

| INST         | OP          | SRC  | DST  | NUM1   | NUM2    |
|--------------|-------------|------|------|--------|---------|
| NoC Scalar   | +=,-=,*=,/= | Addr | Addr | Mask   | Config  |
| NoC Access   | Rd, Wr      | Addr | Addr | Mask   | Const   |
| NoC BCast    | /           | Addr | Addr | Mask   | SrcBank |
| NoC Reduce   | +=,-=,*=,/= | Addr | Addr | Mask   | DstBank |
| NoC Exchange | T+/-,R+/-   | Addr | Addr | Offset | Group   |
| SRAM Write   | /           | Addr | /    | Len    | /       |
| SRAM Comp    | /           | Addr | Addr | Len    | /       |

Within each bank, NoC-related instructions are operated at scalar granularity. From the programming perspective, we view the NoC purely as a computational component in this ISA level, without considering the communication behavior within the NoC. Five NoC-related instructions are designed.

NoC\_Scalar is responsible for once computation in router and NoC\_Access is used to read/write the Curry ALU's registers: the 64-bit Mask is used to indicate whether 64 routers of a channel accept the computation task.

NoC\_Reduce and NoC\_BCast perform DRAM banklevel reduction and broadcasting, using Mask to determine SRAM-PIM macro participation. Both support 4 parallel trees, with DstBank/SrcBank specifying the target/source bank.

NoC\_Exchange differs by allowing both intra-row and inter-bank data exchange, where T and R denote inter-bank and intra-row swaps, +/- indicate inversion, and Offset and Group define swap targets as (x+Offset) % Group. For RoPE, NoC\_Exchange(R-,SrcRow,DstRow,1,2) can be used to express the exchange.

TABLE II PACKET-LEVEL ISA FOR NOC

| Type   | Data | IterNum | Path[0]    | Path[1]      | Path[2] | Path[3]     |
|--------|------|---------|------------|--------------|---------|-------------|
| 4b     | 16b  | 4b      | 12b        | 12b          | 12b     | 12b         |
| Path.X |      | Path.Y  | Path.WrReg | Path.IterTag |         | Path.Opcode |
| 4b     |      | 4b      | 1b         | 1b           |         | 2b          |

Table II shows the packet information at the time of router execution. Where Type is used to indicate the instruction information, currently includes seven types: None, Scalar, Reduce, Exchange, Broadcast, Read, and Write. The Data field contains BF16-formatted payload within the packet. IterNum specifies the iteration count for the computational path, while Path defines the router sequence for each computation step. The control signals include: WrReg for register write-enable in CurryALU, IterTag which triggers dynamic ArgReg updates via IterArg and IterOp after computation.

