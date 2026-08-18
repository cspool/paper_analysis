# Algorithm V.1: Fused rANS Decompression and GEMM Tile Computation

```
Input: Compressed bitstreams B, global rANS decode
        table T, matrices B and C
  Output: Updated output tile C
1 Shared memory:
2 Decode table T˜ (copied from T)
3 Double-buffered decompressed tiles
      A(0), A(1) ∈ RM×K
4 Tile B(k) ∈ R
                  K×N
5 Atomic Flags ready[2] indicating buffer readiness
6 Copy global decode table T into shared memory T˜
7 Initialize ready[0] ← 0, ready[1] ← 0
8 for k = 0 to Ktiles − 1 do
9 b ← k mod 2 // current buffer index
10 p ← 1 − b // previous buffer
11 Warp 0: rANS decompression into A(b)
12 RansDecodeTile(A(b)
                            , B[k], T˜)
13 ready[b] ← 1
14 Warps 1..W: GEMM tile compute
15 Load B(k)
                 from global memory
16 if k = 0 then
17 Wait until ready[b] = 1 // first tile
18 GemmTile(A(b)
                        , B(k)
                             , C)
19 ready[b] ← 0
20 else
21 Wait until ready[p] = 1 // consume
            previous buffer
22 GemmTile(A(p)
                        , B(k)
                             , C)
23 ready[p] ← 0
24 Block-wide synchronize
```

a shared-memory lookup to retrieve the corresponding table entry, and writes the decoded value directly into the sharedmemory.

<sup>25</sup> return C

To maintain correctness of the rANS automaton, each lane independently updates and renormalizes its state by reading additional bits from the compressed stream when the state falls below the renormalization threshold. Because the compressed streams are interleaved across lanes, these renormalization loads are naturally coalesced in global memory, preserving high memory throughput. This warp-level interleaving enables a single SM to decode multiple rANS streams concurrently while maintaining the correctness of the rANS state transitions. As a result, the decoder achieves high parallel efficiency while reconstructing tiles directly into shared memory for immediate consumption by the GEMM kernel.

*4) Tile-Level GEMM Microkernel:* Algorithm [V.3](#page-7-0) then illustrates the complementary consumer stage. Here, the GEMM

Algorithm V.2: Warp-cooperative rANS tile decompression

```
1 Function RansDecodeTile(A, stream, T˜)
2 Initialize rANS state s for each lane
3 for i = 0 to Slane − 1 do
4 x ← s.value mod R // low bits
5 σ, f, c ← T˜[x] // shared-memory lookup
6 w ← DecodeSymbol(σ)
7 Compute (r, c) for symbol index and write
        A[r, c] ← w // A is a shared-memory tile
8 s.value ← f · ⌊s.value/R⌋ + (x − c)
9 while s.value < renorm thresh do
10 u ← Load 32-bit chunk (coalesced)
11 s.value ← (s.value ≪ 32) | u
```

microkernel reads the decompressed weight tile A directly from shared memory and multiplies it with a K ×N activation tile B. Because the weight tile is already resident in shared memory accessible to the warp, all accesses to A[m, k] are single-cycle, eliminating the bandwidth demands and cache thrashing associated with repeatedly loading large weight matrices from global memory.

We build upon CUTLASS to obtain the flexibility required to support the diverse numeric formats and widely used quantization schemes. Moreover, CUTLASS exposes programmable tensor-core tiling, warp scheduling, and memory layouts, allowing us to integrate tile-level ANS decompression directly into the GEMM pipeline, but our proposed design is not tied to CUTLASS itself.

A key consequence of this organization is that decode and compute share the same on-chip shared-memory footprint, so each weight is decoded exactly once and never reloaded. Combined with the producer-consumer scheduling analyzed in Section [V-D,](#page-6-1) the kernel sustains GEMM at full tensor-core speed while keeping the decoder on the critical path only when batch size is small.

## <span id="page-6-1"></span>*D. Pipeline Overlap and the Batch-Size Regime*

The producer (rANS decode) and consumer (tensor-core matrix multiply) execute on physically disjoint pipelines within each streaming multiprocessor: decode warps exercise only the integer and load/store units (probability-table loads, rANS state updates, shared-memory stores into the operand slab), while matrix-multiply warps issue only shared-memory matrix loads and tensor-core multiply–accumulate instructions on the separate tensor pipeline. The warp scheduler co-issues them in the same cycle, and a four-stage shared-memory ring buffer lets decode run several sub-tiles ahead, so each consumer step finds its operands already resident.

The effectiveness of this overlap scales with batch size. Decode cost per sub-tile is approximately constant, dominated by probability-table accesses and renormalization reads; matrixmultiply cost per sub-tile grows with the M-rows processed

# Algorithm V.1: Fused rANS Decompression and GEMM Tile Computation

```
Input: Compressed bitstreams B, global rANS decode
        table T, matrices B and C
  Output: Updated output tile C
1 Shared memory:
2 Decode table T˜ (copied from T)
3 Double-buffered decompressed tiles
      A(0), A(1) ∈ RM×K
4 Tile B(k) ∈ R
                  K×N
5 Atomic Flags ready[2] indicating buffer readiness
6 Copy global decode table T into shared memory T˜
7 Initialize ready[0] ← 0, ready[1] ← 0
8 for k = 0 to Ktiles − 1 do
9 b ← k mod 2 // current buffer index
10 p ← 1 − b // previous buffer
11 Warp 0: rANS decompression into A(b)
12 RansDecodeTile(A(b)
                            , B[k], T˜)
13 ready[b] ← 1
14 Warps 1..W: GEMM tile compute
15 Load B(k)
                 from global memory
16 if k = 0 then
17 Wait until ready[b] = 1 // first tile
18 GemmTile(A(b)
                        , B(k)
                             , C)
19 ready[b] ← 0
20 else
21 Wait until ready[p] = 1 // consume
            previous buffer
22 GemmTile(A(p)
                        , B(k)
                             , C)
23 ready[p] ← 0
24 Block-wide synchronize
```

a shared-memory lookup to retrieve the corresponding table entry, and writes the decoded value directly into the sharedmemory.

<sup>25</sup> return C

To maintain correctness of the rANS automaton, each lane independently updates and renormalizes its state by reading additional bits from the compressed stream when the state falls below the renormalization threshold. Because the compressed streams are interleaved across lanes, these renormalization loads are naturally coalesced in global memory, preserving high memory throughput. This warp-level interleaving enables a single SM to decode multiple rANS streams concurrently while maintaining the correctness of the rANS state transitions. As a result, the decoder achieves high parallel efficiency while reconstructing tiles directly into shared memory for immediate consumption by the GEMM kernel.

*4) Tile-Level GEMM Microkernel:* Algorithm [V.3](#page-7-0) then illustrates the complementary consumer stage. Here, the GEMM

Algorithm V.2: Warp-cooperative rANS tile decompression

```
1 Function RansDecodeTile(A, stream, T˜)
2 Initialize rANS state s for each lane
3 for i = 0 to Slane − 1 do
4 x ← s.value mod R // low bits
5 σ, f, c ← T˜[x] // shared-memory lookup
6 w ← DecodeSymbol(σ)
7 Compute (r, c) for symbol index and write
        A[r, c] ← w // A is a shared-memory tile
8 s.value ← f · ⌊s.value/R⌋ + (x − c)
9 while s.value < renorm thresh do
10 u ← Load 32-bit chunk (coalesced)
11 s.value ← (s.value ≪ 32) | u
```

microkernel reads the decompressed weight tile A directly from shared memory and multiplies it with a K ×N activation tile B. Because the weight tile is already resident in shared memory accessible to the warp, all accesses to A[m, k] are single-cycle, eliminating the bandwidth demands and cache thrashing associated with repeatedly loading large weight matrices from global memory.

We build upon CUTLASS to obtain the flexibility required to support the diverse numeric formats and widely used quantization schemes. Moreover, CUTLASS exposes programmable tensor-core tiling, warp scheduling, and memory layouts, allowing us to integrate tile-level ANS decompression directly into the GEMM pipeline, but our proposed design is not tied to CUTLASS itself.

A key consequence of this organization is that decode and compute share the same on-chip shared-memory footprint, so each weight is decoded exactly once and never reloaded. Combined with the producer-consumer scheduling analyzed in Section [V-D,](#page-6-1) the kernel sustains GEMM at full tensor-core speed while keeping the decoder on the critical path only when batch size is small.

## <span id="page-6-1"></span>*D. Pipeline Overlap and the Batch-Size Regime*

The producer (rANS decode) and consumer (tensor-core matrix multiply) execute on physically disjoint pipelines within each streaming multiprocessor: decode warps exercise only the integer and load/store units (probability-table loads, rANS state updates, shared-memory stores into the operand slab), while matrix-multiply warps issue only shared-memory matrix loads and tensor-core multiply–accumulate instructions on the separate tensor pipeline. The warp scheduler co-issues them in the same cycle, and a four-stage shared-memory ring buffer lets decode run several sub-tiles ahead, so each consumer step finds its operands already resident.

The effectiveness of this overlap scales with batch size. Decode cost per sub-tile is approximately constant, dominated by probability-table accesses and renormalization reads; matrixmultiply cost per sub-tile grows with the M-rows processed

