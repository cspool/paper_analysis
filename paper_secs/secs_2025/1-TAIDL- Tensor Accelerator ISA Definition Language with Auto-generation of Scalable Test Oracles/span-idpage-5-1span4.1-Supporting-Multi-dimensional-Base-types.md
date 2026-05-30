# <span id="page-5-1"></span>4.1 Supporting Multi-dimensional Base-types

We observe that tensor accelerators perform computations on multidimensional data present in tensor buffers. Without support for multi-dimensional base-types, the semantics of every instruction have to reshape the data before performing any computation over it. Thus, we design TAIDL to support multi-dimensional base-types, making the instruction semantics compact and easy to understand by avoiding complex address computations.

```
1 # Data Model: AMX Tile and AVX-512 Register file
2 [tiles] (8) (16x64xi8); # 8 AMX Tile registers
3 [zmm] (32) (16xf32); # 32 AVX-512 registers
```

Figure 7: TAIDL definition of Intel AMX & AVX-512 registers.

Figure 7 models Intel AMX tiles and AVX-512 registers in TAIDL. Each tile can hold up to 16 rows with up to 64 bytes per row. Instead of representing a tile as a buffer of 1024 bytes, TAIDL allows it to be represented as a 2-dimensional base type of 16x64xi8 (line 2).

#### <span id="page-5-2"></span>4.2 Supporting Multi-dimensional Addressing

Several data buffers found in tensor accelerators are partitioned into parallelly-accessed banks and also support strided accesses. Keeping this in mind, we design TAIDL to support multi-dimensional addressing as an extension to commonly observed 1-dimensional addresses. This simplifies address computation for data accesses.

```
1 # Data Model: MXU FIFO Buffers for TPUv2
2 [MXU_in_fifo] (1,256) (128xf32);
3 # Data Model: MXU FIFO Buffers for TPUv3
4 [MXU_in_fifo] (2,256) (128xf32);
5 # Instruction Semantics using MXU
6 %In:1x256x128xf32 <- MXU_in_fifo[mxu_id, 0];</pre>
```

Figure 8: TAIDL definition of TPU MXU FIFO buffers.

Figure [8](#page-5-6) shows a snippet of the TAIDL definition of the MXU buffers for TPUv2 (line 2) and TPUv3 (line 4). The TAIDL definition takes advantage of the multi-dimensional addressing construct by adding another dimension to the MXU FIFO buffers, representing the MXU id (analogous to the batch dimension in batch processing).

