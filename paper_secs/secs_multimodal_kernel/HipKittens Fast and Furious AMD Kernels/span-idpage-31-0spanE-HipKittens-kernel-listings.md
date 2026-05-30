# <span id="page-31-0"></span>E HipKittens kernel listings

This section demonstrates HipKittens kernel examples and discusses the algorithmic details of our kernel implementations.

### E.1 Matrix Multiply

The BF16 GEMM kernel (Section [E.3\)](#page-38-0) decomposes the problem into computing a 256 × 256 output tile per thread block (denoted by BLOCK\_SIZE). In the prologue, the kernel pre-loads the A and B input matrices from global to shared memory. The kernel inserts a conditional barrier to stall half the waves (one wave per SIMD) while the other half begins performing additional loads. When this leader wavegroup finishes its additional loads, it unblocks the follower wavegroup through the s\_barrier invocation. Thereafter, the two wavegroups alternate between the compute and memory clusters shown in the hotloop, where the end of a cluster is always demarked by an s\_barrier. This represents the 8-wave ping pong kernel schedule introduced in Section [3.3.](#page-6-1)

For the MI325X version of the kernel, we maintain the same 8-wave structure, however the hardware only has 65 KB of shared memory so we cannot double buffer in shared memory. Instead, we double buffer using the register file. We do not use the direct HBM to LDS buffer loads, and instead load from HBM to a register buffer, while the waves compute MFMAs on the previously stored register tiles. When the compute completes, the data from the register buffers gets stored down to shared memory using a ds\_write.

```
constexpr int BLOCK_SIZE
                                          = 256;
     constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2;
 3
     constexpr int K_STEP
                                           = 64;
     constexpr int WARPS_M
                                           = 2;
     constexpr int WARPS_N
                                           = 4;
                                           = BLOCK_SIZE / WARPS_M;
     constexpr int REG_BLOCK_M
                                           = BLOCK_SIZE / WARPS_N;
     constexpr int REG_BLOCK_N
     constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
     constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;
10
     constexpr int DOT_SLICE
                                           = 32;
11
     __global__ void GEMM_BF16(const micro_globals g) {
13
         // setup
          extern __shared__ alignment_dummy __shm[];
shared_allocator al((int*)&__shm[0]);
15
          using ST_A = st_bf <HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;\nusing ST_B = st_bf <HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;
16
17
          ST_A (&As)[2][2] = al.allocate < ST_A, 2, 2>();
18
19
          ST_B (\&Bs)[2][2] = al.allocate < ST_B, 2, 2>();
20
21
          \label{eq:rt_bf} \verb|rt_bf| < \verb|HALF_REG_BLOCK_M|, K_STEP|, row_l|, rt_16x32_s > A_tile; \\
          rt_bf < HALF_REG_BLOCK_N, K_STEP, row_l, rt_16x32_s > B_tile_0; rt_bf < HALF_REG_BLOCK_N, K_STEP, row_l, rt_16x32_s > B_tile_1;
23
24
          rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];
25
          zero(C accum[0][0]):
26
          zero(C_accum[0][1]);
27
          zero(C_accum[1][0]);
28
         zero(C_accum[1][1]);
29
30
          // L2 and LLC Cache swizzling
          int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
31
32
33
          const int NUM_WGS = gridDim.x * gridDim.y;
          const int WGM = 8:
34
          // Swizzle chiplet so that wgids are in the same \mbox{XCD}.
35
          wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
36
          // Swizzle for better L2 within the same XCD.
          const int num_pid_m = ceil_div(M, BLOCK_SIZE);
const int num_pid_n = ceil_div(N, BLOCK_SIZE);
37
39
          const int num_wgid_in_group = WGM * num_pid_n;
          int group_id = wgid / num_wgid_in_group;\nint first_pid_m = group_id * WGM;\nint group_size_m = min(num_pid_m - first_pid_m, WGM);
40
41
42
43
          int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
44
          int pid_n = (wgid % num_wgid_in_group) / group_size_m;
45
          // Assign the tile's row/column based on the pid_m and pid_n.
46
          int row = pid_m;
          int col = pid_n;
47
48
49
          const int warp_id = kittens::warpid();
50
          const int warp_row = warp_id / 4;
          const int warp_col = warp_id % 4;
52
          const int num_tiles = K / K_STEP;
53
54
          int tic = 0;
         int toc = 1;
56
         // preload
         G::load(Bs[tic][0], g.b, {0, 0, col*2, 0});
          G::load(As[tic][0], g.a, {0, 0, row*2, 0});
G::load(Bs[tic][1], g.b, {0, 0, col*2 + 1, 0});
59
60
61
          G::load(As[tic][1], g.a, {0, 0, row*2 + 1, 0});
62
63
          // conditional stagger
64
         if (warp_row == 1) {
65
               __builtin_amdgcn_s_barrier();
66
67
68
          asm volatile("s_waitcnt vmcnt(4)");
69
          __builtin_amdgcn_s_barrier();
70
71
72
73
74
75
76
          // preload
          G::load(Bs[toc][0], g.b, {0, 0, col*2, 1});
G::load(As[toc][0], g.a, {0, 0, row*2, 1});
G::load(Bs[toc][1], g.b, {0, 0, col*2 + 1, 1});
          asm volatile("s_waitcnt vmcnt(6)"):
          __builtin_amdgcn_s_barrier();
```

Figure 21: HK BF16 GEMM, which is competitive with AITER on CDNA4.

```
#pragma unroll
                        for (int tile = 0: tile < num tiles - 2: ++tile, tic^=1, toc^=1) {
  3
                                    auto st_subtile_b = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[tic][0], {warp_col, 0});
  4
  5
                                   load(B_tile_0, st_subtile_b);
  6
7
                                    auto st_subtile_a = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
                                   load(A_tile, st_subtile_a);
  8
                                   G::load(As[toc][1], g.a, {0, 0, row*2 + 1, tile + 1});
  9
                                   asm volatile("s_waitcnt lgkmcnt(8)");
                                    __builtin_amdgcn_s_barrier();
10
11
                                  asm volatile("s_waitcnt lgkmcnt(0)");
12
13
                                     __builtin_amdgcn_s_setprio(1);
14
                                   mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
15
                                   __builtin_amdgcn_s_setprio(0);
16
                                   __builtin_amdgcn_s_barrier();
17
                                    __builtin_amdgcn_sched_barrier(0);
18
19
                                   st_subtile_b = subtile_inplace < HALF_REG_BLOCK_N, K_STEP > (Bs[tic][1], {warp_col, 0});
                                   load(B_tile_1, st_subtile_b);
20
21
                                   G::load(Bs[tic][0], g.b, {0, 0, col*2, tile + 2});
22
23
                                   __builtin_amdgcn_s_barrier();
24
                                  asm volatile("s_waitcnt lgkmcnt(0)");
25
                                     __builtin_amdgcn_s_setprio(1);
26
                                   mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
27
                                    __builtin_amdgcn_s_setprio(0);
28
                                    __builtin_amdgcn_s_barrier();
29
30
                                   st_subtile_a = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
                                   load(A_tile, st_subtile_a);
32
                                   G::load(As[tic][0], g.a, {0, 0, row*2, tile + 2});
33
                                    __builtin_amdgcn_s_barrier();
34
35
                                   asm volatile("s_waitcnt lgkmcnt(0)");
36
                                    __builtin_amdgcn_s_setprio(1);
37
                                   mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
38
                                   __builtin_amdgcn_s_setprio(0);
39
                                    __builtin_amdgcn_s_barrier();
40
                                   __builtin_amdgcn_sched_barrier(0);
41
42
                                   G::load(Bs[tic][1], g.b, {0, col*2 + 1, tile + 2});
43
                                  asm volatile("s_waitcnt vmcnt(6)");
44
                                    __builtin_amdgcn_s_barrier();
45
46
                                      __builtin_amdgcn_s_setprio(1);
                                   mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
48
                                    __builtin_amdgcn_s_setprio(0);
                                    __builtin_amdgcn_s_barrier();
49
50
51
52
                       // Epilogue not shown
53
54
                        if (warp_row == 0) {
55
                                    __builtin_amdgcn_s_barrier();
56
58
                        // store results
                       store(g.c, C_accum[0][0], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + warp_col}); store(g.c, C_accum[0][1], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + WARPS_N +
59
60
                                    → warp_col});
61
                        \texttt{store}(\texttt{g.c.}, \texttt{C\_accum[1][0]}, \texttt{\{0, 0, (row * 2) * WARPS\_M + WARPS\_M + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_
                                      → warp_col});
                        \texttt{store}(\texttt{g.c, C\_accum[1][1], \{0, 0, (row * 2) * WARPS\_M + WARPS\_M + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_row, col * 2 * WARPS\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + warp\_N + war
62

→ WARPS_N + warp_col});
63
          }
```

#### E.2 Fused Dropout + Residual + Layernorm

A very simple HIPKITTENS kernel that processes a chunk of vectors along the sequence dimension per thread block. This kernel listing demonstrates HK operators and vectors, which resemble those in PyTorch (e.g., sum, add, mul, div).

```
__global__ void fused_layernorm(const norm_globals g) {
 2
3
         auto warpid = kittens::warpid():
4
         const int batch = blockIdx.y;
const int seq_start = blockIdx.x*g.n_per_tile;
5
6
7
         constexpr int d_model = 128;
9
         rv_naive < bf16, d_model > residual_s_reg, x_s_reg, norm_weight_s_reg, norm_bias_s_reg;
10
         load(x_s_reg, g.x, {0, batch, seq_start + warpid, 0});
         asm volatile(
11
         load(residual_s_reg, g.residual, {0, batch, seq_start + warpid, 0});
12
13
         bf16 mean = __float2bfloat16(0.0f);
14
         bf16 var = __float2bfloat16(0.0f);
         if constexpr (DROPOUT_P > 0.0f) +
15
16
              dropout_mask(x_s_reg, DROPOUT_P);
17
18
         if constexpr (DROPOUT_P > 0.0f) {
             constexpr float scale = 1.0f / (1.0f - DROPOUT_P);
mul(x_s_reg, x_s_reg, __float2bfloat16(scale));
19
20
21
22
         asm volatile("s_waitcnt vmcnt(0)");
23
         load(norm\_weight\_s\_reg\,,\ g.norm\_weight\,,\ \{0\,,0\,,0\,,0\})\,;
24
25
         load(norm_bias_s_reg, g.norm_bias, {0,0,0,0});
         add(residual_s_reg, residual_s_reg, x_s_reg);
26
         store(g.o_resid, residual_s_reg, {0, batch, seq_start + warpid, 0});
27
28
         // mean and variance
29
         sum(mean, residual_s_reg);
30
         constexpr float dim_scale = 1.0f / d_model;
31
         mean = mean * __float2bfloat16(dim_scale);
32
         sub(residual_s_reg, residual_s_reg, mean);
33
         mul(x_s_reg, residual_s_reg, residual_s_reg);
34
         sum(var, x_s_reg);
35
         var = var * __float2bfloat16(dim_scale);
36
         var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));
37
38
         // compute norm
39
         div(residual_s_reg, residual_s_reg, var);
         asm volatile("s_waitcnt vmcnt(0)");
         mul(residual_s_reg, residual_s_reg, norm_weight_s_reg);
41
         add(residual_s_reg, residual_s_reg, norm_bias_s_reg);
store(g.o, residual_s_reg, {0, batch, seq_start+warpid, 0});
43
```

Figure 22: Fused Dropout + Residual + Layernorm kernel outperforming torch.compile.

## E.3 Attention

The HipKittens attention kernel uses an 8-wave ping pong schedule. Each wave computes a 32 × 128 tile of the output for a single head and batch. In the prologue, all eight waves first collaboratively load tiles of keys and values, and their own personal tiles of queries. The threads perform the initial query-key matrix multiply and first half of softmax. Then the kernel uses a conditional barrier to stall half the waves (one wave per SIMD). The leader wavegroup proceeds ahead, loading the next tiles of keys and values and upon completion, unlocks the follower wavegroup. In the hotloop, the two wavegroups alternate between compute clusters (each involving matrix multiplies and vector operations) and load clusters (involving global to shared and shared to register loads).

In compute clusters, the compiler interleaves vector ops and matrix ops. We can also use sched\_barrier hints to guide the LLVM compiler to interleave in custom patterns.

```
template < int D>
2
     __global__ void attend_ker(const attn_globals <D> g) {
3
         extern __shared__ alignment_dummy __shm[];
         shared_allocator al((int*)&__shm[0]);
 5
         st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, 

6
 7
         st_bf < KV_BLOCK_SIZE, ATTN_D, st_8x32_s > (&v_smem)[2] = al.allocate < st_bf < KV_BLOCK_SIZE, ATTN_D,
              \hookrightarrow st_8x32_s>, 2>();
8
9
         const int head_idx = (blockIdx.x % GROUP_SIZE) * GROUP_SIZE + (blockIdx.x / GROUP_SIZE);
         const int batch_idx = blockIdx.z;
10
         const int head_idx_kv = head_idx / GROUP_SIZE;
11
         const int block_tile_idx = blockIdx.y;
12
         const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
const int stagger = warpid() / 4;
13
14
15
         const int num_tiles = ATTN_N / KV_BLOCK_SIZE;
16
         constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089
17
              → f •
18
         // Initialize all of the register tiles.
19
         qo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
20
21
         qo_tile_transposed <D, bf16 > q_reg_transposed;
         kv_tile <D, bf16 > k_reg;
22
23
         kv_tile_transposed <D, bf16 > k_reg_transposed;
24
25
         kv_tile < D, bf16, col_1, rt_32x32_s > v_reg;
26
         qo_tile_transposed <D, float, col_1, rt_32x32_s > o_reg; // Output tile.
27
         attn_tile <D, float, col_1, rt_32x32_s> att_block[2]; // attention tile, in float.
28
         attn_tile <D, bf16, col_l, rt_32x32_s > att_block_bf16;
20
         typename attn_tile <D, float, col_1, rt_32x32_s>::row_vec max_vec, norm_vec, max_vec_prev;
30
31
         \label{eq:G:cond} G::load<1, \  \, \mbox{false}>(\mbox{k\_smem}\left[0\right], \ \mbox{g.Kg, } \{\mbox{batch\_idx, 0, head\_idx\_kv, 0}\});
32
         __builtin_amdgcn_s_waitcnt(0);
33
         __builtin_amdgcn_sched_barrier(0);
34
         __builtin_amdgcn_s_barrier();
35
36
         qo_tile <D, float > q_reg_fl;
         load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
37
         mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE); // Use sqrtf for clarity
38
39
         copy(q_reg, q_reg_fl);
40
         swap_layout_and_transpose(q_reg_transposed, q_reg);
41
42
        zero(o_reg);
         zero(norm_vec);
43
44
        neg_infty(max_vec_prev);
         // All warps then collaboratively load in the first slice of V (VO) and the second slice of K (K1)
46

→ into shared memory

47
         G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0});
         // All warps then load in the first slice of K (KO)
49
         G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0});
         load(k_reg, k_smem[0]);
asm volatile("s_waitcnt lgkmcnt(0)");
asm volatile("s_waitcnt vmcnt(2)");
50
51
52
         __builtin_amdgcn_sched_barrier(0);
53
54
         __builtin_amdgcn_s_barrier();
55
56
         // each warp performs QKO
57
        zero(att block[0]):
         {\tt swap\_layout\_and\_transpose(k\_reg\_transposed, k\_reg);}
58
59
         mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
60
61
         // each warp performs a partial softmax of QKO
         col_max(max_vec, att_block[0]);
62
63
         sub_col(att_block[0], att_block[0], max_vec);
64
         exp2(att_block[0], att_block[0]);
65
         sched_barrier_pairs <8, 6, 1>();
66
         sched_barrier_exp_pairs < 8, 4, 1 > ();
67
68
         // conditional stagger
         if (stagger) {
70
             __builtin_amdgcn_sched_barrier(0);
71
72
             __builtin_amdgcn_s_barrier();
         }
```

```
// All warps then load in the second slice of K (K1)
 2
          load(k_reg, k_smem[1]);
         // All warps then collaboratively load in the third slice of K (K2) into shared memory
G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0});
 3
 4
 5
          // All warps then collaboratively load in the second slice of V (V1) into shared memory
         G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0});
asm volatile("s_waitcnt lgkmcnt(0)");
asm volatile("s_waitcnt vmcnt(4)");
 6
 8
         __builtin_amdgcn_sched_barrier(0);
 9
10
          __builtin_amdgcn_s_barrier();
11
12
13
         // hot loop
         for (int j = 3; j < num_tiles - 1; j += 2) {</pre>
15
             // Cluster 0:
                  QK1
16
              zero(att_block[1]);
17
              swap_layout_and_transpose(k_reg_transposed, k_reg);
18
19
              mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
20
                       Finish softmax for QKO
21
              sub(max_vec_prev, max_vec_prev, max_vec);
22
              exp2(max_vec_prev, max_vec_prev);
23
              mul(norm_vec, norm_vec, max_vec_prev);
24
              col_sum(norm_vec, att_block[0], norm_vec);
25
              copy(att_block_bf16, att_block[0]);
26
              sched_barrier_pairs <16, 3, 2>();
27
              __builtin_amdgcn_sched_barrier(0);
28
               __builtin_amdgcn_s_barrier();
29
              __builtin_amdgcn_sched_barrier(0);
30
31
              // Cluster 1:
32
                       Load K3 into shared
33
              \label{eq:G:cond} G::load<1, \  \, \mbox{false}>(\mbox{k\_smem}\,[1]\,, \ \mbox{g.Kg, } \{\mbox{batch\_idx}\,, \ \mbox{j, head\_idx\_kv}\,, \ \mbox{0}\})\,;
34
                      Load VO into registers
              11
35
              load(v_reg, v_smem[0]);
36
              asm volatile("s_waitcnt lgkmcnt(0)");
asm volatile("s_waitcnt vmcnt(4)");
37
38
              __builtin_amdgcn_sched_barrier(0);
39
              __builtin_amdgcn_s_barrier();
40
              __builtin_amdgcn_sched_barrier(0);
41
42
              // Cluster 2:
43
              //
                      AOVO
44
              __builtin_amdgcn_s_setprio(1);
45
              mul_col(o_reg, o_reg, max_vec_prev);
46
              __builtin_amdgcn_sched_barrier(0);
              mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
// Partial softmax for QK1
47
48
49
              copy(max_vec_prev, max_vec);
              col_max(max_vec, att_block[1], max_vec);
50
51
              sub_col(att_block[1], att_block[1], max_vec);
52
              exp2(att_block[1], att_block[1]);
53
              sched_barrier_pairs <8, 6, 3>();
54
              sched_barrier_exp_pairs <8, 4, 3>();
              __builtin_amdgcn_s_setprio(0);
56
              __builtin_amdgcn_sched_barrier(0);
57
              __builtin_amdgcn_s_barrier();
58
              __builtin_amdgcn_sched_barrier(0);
59
60
              // Cluster 3:
61
                       Load V2 into shared
62
              G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0});
                       Load K2 into registers
63
64
              load(k_reg, k_smem[0]);
65
              asm volatile("s_waitcnt lgkmcnt(0)");
asm volatile("s_waitcnt vmcnt(4)");
66
67
              __builtin_amdgcn_sched_barrier(0);
              __builtin_amdgcn_s_barrier();
              __builtin_amdgcn_sched_barrier(0);
```

```
// Cluster 4:
2
                      QK2
3
             zero(att_block[0]);
 4
             swap_layout_and_transpose(k_reg_transposed, k_reg);
 5
             mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
 6
7
                      Finish softmax for QK1
             sub(max_vec_prev, max_vec_prev, max_vec);
             exp2(max_vec_prev, max_vec_prev);
9
             mul(norm_vec, norm_vec, max_vec_prev);
10
             col_sum(norm_vec, att_block[1], norm_vec);
11
             copy(att_block_bf16, att_block[1]);
12
             sched_barrier_pairs <16, 3, 4>();
13
             __builtin_amdgcn_sched_barrier(0);
             __builtin_amdgcn_s_barrier();
14
15
             __builtin_amdgcn_sched_barrier(0);
16
17
             // Cluster 5:
                      Load K4 into shared
19
             \label{eq:G:cond} G::load<1, \  \, false>(k\_smem[0], \ g.Kg, \ \{batch\_idx, \ j \ + \ 1, \ head\_idx\_kv \,, \ 0\});
                     Load V1 into registers
21
             load(v_reg, v_smem[1]);
22
             asm volatile("s_waitcnt lgkmcnt(0)");
asm volatile("s_waitcnt vmcnt(4)");
23
24
             __builtin_amdgcn_sched_barrier(0);
25
             __builtin_amdgcn_s_barrier();
26
             __builtin_amdgcn_sched_barrier(0);
27
             // Cluster 6:
29
             //
                     A1V1
30
              _builtin_amdgcn_s_setprio(1);
31
             mul_col(o_reg, o_reg, max_vec_prev);
32
              __builtin_amdgcn_sched_barrier(0);
33
             mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
// Partial softmax for QK2
34
35
             copy(max_vec_prev, max_vec);
36
             col_max(max_vec, att_block[0], max_vec);
37
             sub_col(att_block[0], att_block[0], max_vec);
38
             exp2(att_block[0], att_block[0]);
39
             sched_barrier_pairs < 8, 6, 5 > ();
40
             sched_barrier_exp_pairs <8, 4, 5>();
41
             __builtin_amdgcn_s_setprio(0);
42
             __builtin_amdgcn_sched_barrier(0);
43
             __builtin_amdgcn_s_barrier();
44
             __builtin_amdgcn_sched_barrier(0);
45
46
             // Cluster 7:
                     Load V3 into shared
47
48
             G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0});
                     Load K3 into registers
49
50
             load(k_reg, k_smem[1]);
             asm volatile("s_waitcnt lgkmcnt(0)");
asm volatile("s_waitcnt vmcnt(4)");
51
52
53
             __builtin_amdgcn_sched_barrier(0);
54
             __builtin_amdgcn_s_barrier();
55
             __builtin_amdgcn_sched_barrier(0);
56
58
         // Epilogue not shown
59
60
         // Conclusion
61
         if (!stagger) {
62
             __builtin_amdgcn_s_barrier();
63
64
65
         qo_tile <D, float, row_l, rt_32x32_s > o_reg_transposed;
66
         swap_layout_and_transpose(o_reg_transposed, o_reg);
67
         store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});
68
69
         // multiply by ln(2)
70
71
         mul(max_vec, max_vec, 0.69314718056f);
         log(norm_vec, norm_vec);
72
         add(norm_vec, norm_vec, max_vec);
73
74
         \verb|store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx})|;\\
```

<span id="page-38-0"></span>Figure 23: HIPKITTENS non-causal attention forwards kernel that competes with the assembly kernel provided in AMD's AITER library.

