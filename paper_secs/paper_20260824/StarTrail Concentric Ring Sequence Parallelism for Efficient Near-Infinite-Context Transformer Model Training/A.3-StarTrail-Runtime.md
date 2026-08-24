# A.3 StarTrail Runtime

StarTrail is written in PyTorch[\[29\]](#page-11-11) and uses the PyTorch torch.autograd.function and NCCL[\[28\]](#page-11-12) backend for forward and backward implementation. StarTrail also employs multiple techniques during runtime to improve its overall training efficiency.

Ingetrate Flash Attention. The StarTrail attention mechanism involves multiple iterations that loop over Keys and Values, with each iteration still using traditional self-attention with corresponding Query, Key, and Value (QKV). This approach enables StarTrail to incorporate flash attention effectively, extending its capability by preserving intermediate states across iterations. Additionally,

### <span id="page-21-0"></span>**Algorithm 3** get\_P2P\_config()

**Require:** inter-team rank  $\mathbf{r}_t$ , intra-team rank  $\mathbf{r}_a$ , inter-team dimension  $\mathbf{d}_t$ , intra-team dimension  $\mathbf{d}_a$ 

- 1: team group size =  $\mathbf{d}_t / \mathbf{d}_a$
- 2: self team group rank =  $\frac{r_t}{team \ group \ size}$
- 3: next team in group =  $(r_t + 1)\%$  team group size + team group size × self team group rank
- 4: last team in group =  $(r_t 1)\%$  team group size + team group size × self team group rank
- 5: next device global rank =  $r_a$ + next team in group  $\times d_a$
- 6: last device global rank =  $r_a$  + last team in group  $\times d_a$
- 7: return next device global rank, last device global rank

<span id="page-21-1"></span>Table 2: Supported Sequence Length of Ring Attention and StarTrail on one Nvidia A100 80GB GPU

| Supported Seq Len on one 80GB A100 GPU (K Tokens) |        |                |                           |  |  |  |  |
|---------------------------------------------------|--------|----------------|---------------------------|--|--|--|--|
| Model Size                                        | Length | Ring Attention | StarTrail                 |  |  |  |  |
|                                                   | 128    | <b>√</b>       | <b>√</b>                  |  |  |  |  |
| 3B                                                | 256    | <b>√</b>       | $\overline{\hspace{1cm}}$ |  |  |  |  |
|                                                   | 512    | <b>√</b>       | <b>√</b>                  |  |  |  |  |
|                                                   | 128    | ✓              | <b>√</b>                  |  |  |  |  |
| 7B                                                | 256    | <b>√</b>       | $\overline{\hspace{1cm}}$ |  |  |  |  |
|                                                   | 512    | Х              | Х                         |  |  |  |  |
|                                                   | 128    | ✓              | <b>√</b>                  |  |  |  |  |
| 13B                                               | 256    | Х              | X                         |  |  |  |  |
|                                                   | 512    | X              | Х                         |  |  |  |  |

StarTrail enhances the efficiency of the forward process with the help of torch JIT to fuse kernels aside from flash attention.

**Overlap communication with computing.** In StarTrail attention, P2P communication and self-attention computing are interleaved across iterations, each incurring considerable time. To mitigate this, StarTrail employs a double buffering technique to asynchronously execute communication and computing kernels, effectively overlapping these processes and enhancing GPU utilization.

**Save recomputation with checkpoints.** StarTrail adopts the checkpointing strategy introduced by DistFlashAttn[20], placing checkpoints at the end of the self-attention phase rather than the FFN of each transformer layer. This checkpoint placement effectively obviates the need to recompute the self-attention forward process during the backward pass, avoiding redundant attention computation.

