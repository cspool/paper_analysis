# Algorithm 2 SRAMFFN-BACKWARD-TRITON(DQ, DR)

**Require:** saved Q, K, U, V, R; incoming  $dS \in \mathbb{R}^{B \times H \times L \times d_h}$ 

**Strategy:** Parallelize over batch, head, and sequence blocks. Inner loops iterate over subnetworks and intermediate dimension blocks.

### **Grid/Meta** as in Alg 1

```
1: s_0 \leftarrow \text{pid0} \cdot \text{BLOCK\_SEQ}; \quad h \leftarrow \text{pid1}; \quad b \leftarrow \text{pid2}
 2: Q_{\text{blk}} \leftarrow Q[b, h, s_0: s_0 + \text{BLOCK\_SEQ}, :]; \quad dS_{\text{blk}} \leftarrow dS[b, h, s_0: s_0 + \text{BLOCK\_SEQ}, :] 3: dQ_{\text{acc}} \leftarrow \mathbf{0}^{\text{BLOCK\_SEQ} \times d_h}; \quad dR_{\text{rows}} \leftarrow \mathbf{0}^{\text{BLOCK\_SEQ} \times 1}
 4: for e = 0 to E - 1 do
 5:
                R_{\text{rows}} \leftarrow R[b, h, s_0 : s_0 + \text{BLOCK\_SEQ}, e]
               for m=0 to d_e-1 step <code>BLOCK_INTER</code> do
 6:
                       K_{\text{tile}} \leftarrow K[h, e, m:m + \text{BLOCK\_INTER}, :];
 7:
                       U_{\text{tile}} \leftarrow U[\dot{h}, e, m: m + \text{BLOCK\_INTER}, :]_{\dot{\underline{:}}}
 8:
                        V_{\mathrm{tile}}^{\top} \leftarrow V[h, e, m : m + \mathtt{BLOCK\_INTER}, :]
 9:
                       \begin{array}{l} M \leftarrow Q_{\text{blk}} \cdot K_{\text{tile}}^{\top}; \quad N \leftarrow Q_{\text{blk}} \cdot U_{\text{tile}}^{\top} \\ \text{sig} \leftarrow \text{sigmoid}(M); \quad \text{SiLU}(M) \leftarrow M \odot \text{sig} \end{array}
10:
11:
                       d\tilde{A} \leftarrow dS_{\text{blk}} \cdot V_{\text{tile}}^{\top}
12:
                       dR_{\text{rows}} += \sum_{\text{cols}} (dA \odot \text{SiLU}(M) \odot N)
13:
                       dM \leftarrow (dA \odot (R_{\text{rows}} \odot N)) \odot (\text{sigmoid}(M) + M \cdot \text{sigmoid}(M) \cdot (1 - \text{sigmoid}(M)))
14:
                       dN \leftarrow (dA \odot \text{SiLU}(M)) \odot R_{\text{rows}}
15:
                       dQ_{\rm acc} + = dM \cdot K_{\rm tile} + dN \cdot U_{\rm tile}
16:
17:
18:
               dR[b, h, s_0: s_0 + \text{BLOCK\_SEQ}, e] \leftarrow dR_{\text{rows}}
19: end for
20: dQ[b, h, s_0: s_0 + \text{BLOCK\_SEQ}, :] \leftarrow dQ_{\text{acc}}
```

### Algorithm 3 SRAMFFN-BACKWARD-TRITON(DK, DU, DV)

```
Require: saved Q, K, U, V, R; dS
                 Grid: pid0\leftarrow e \cdot (d_e/\text{BLOCK\_INTER}) + |m/\text{BLOCK\_INTER}|, pid1\leftarrow h
     1: decode(e, m) from pid0
    2: K_{\text{tile}} \leftarrow K[h, e, m:m + \text{BLOCK\_INTER}, :]; \ U_{\text{tile}} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :]; \ V_{\text{tile}}^{\top} \leftarrow U[h, e, m:m + \text{BLOCK\_INTER}, :];
    V[h,e,m:m+\texttt{BLOCK\_INTER},:]^\top \\ 3: \ dK_{\text{acc}}, dU_{\text{acc}}, dV_{\text{acc}} \leftarrow \mathbf{0}^{\texttt{BLOCK\_INTER} \times d_h}
    4: for b = 0 to B-1 do
                                  for s_0=0 to L{-}1 step <code>BLOCK_SEQ</code> do
                                                     Q_{\text{blk}} \leftarrow Q[b, h, s_0 : s_0 + \text{BLOCK\_SEQ}, :];
    6:
                                                     R_{\text{rows}} \leftarrow R[b, h, s_0 : s_0 + \text{BLOCK\_SEQ}, e];
    7:
                                                     dS_{\text{blk}} \leftarrow d\tilde{S}[b, h, s_0: s_0 + \text{BLOCK\_SEQ}, :]
    8:
                                                   M \leftarrow Q_{\text{blk}} \cdot K_{\text{tile}}^{\top}; \quad N \leftarrow Q_{\text{blk}} \cdot U_{\text{tile}}^{\top}
    9:
                                                   \operatorname{sig} \leftarrow \operatorname{sigmoid}(M); \quad \operatorname{SiLU}(M) \leftarrow M \odot \operatorname{sig}; \quad N \leftarrow R_{\operatorname{rows}} \odot N
10:
                                                    A^{\top} \leftarrow (\operatorname{SiLU}(M) \odot N)^{\top}; \quad dA \leftarrow dS_{\operatorname{blk}} \cdot V_{\operatorname{tile}}^{\top}
11:
                                                    dM \leftarrow (dA \odot N) \odot (\operatorname{sigmoid}(M) + M \cdot \operatorname{sigmoid}(M) \cdot (1 - \operatorname{sigmoid}(M)));
12:
                                                   \begin{array}{l} dN \leftarrow (dA \odot \text{SiLU}(\dot{M})) \odot R_{\text{rows}} \\ dV_{\text{acc}} += A^{\top} \cdot \underline{d}S_{\text{blk}} \end{array}
13:
14:
                                                   dK_{\rm acc} += dM^{\top} \cdot Q_{\rm blk}; \quad dU_{\rm acc} += dN^{\top} \cdot Q_{\rm blk}
15:
                                  end for
16:
17: end for
18: dK[h,e,m:m+\texttt{BLOCK\_INTER},:] \leftarrow dK_{acc}; dU[h,e,m:m+\texttt{BLOCK\_INTER},:] \leftarrow dU_{acc};
                  dV[h, e, m:m+BLOCK\_INTER, :] \leftarrow dV_{acc}
```

### <span id="page-13-0"></span>B HOPPER/THUNDERKITTENS PSEUDOCODE FOR SRAMFFN

#### Algorithm 4 SRAMFFN-FORWARD-TK (HOPPER) **Require:** $Q \in \mathbb{R}^{B \times H \times L \times d_h}$ , $K, U, V \in \mathbb{R}^{H \times E \times d_e \times d_h}$ , $R \in \mathbb{R}^{B \times H \times E \times L}$ Meta: BLOCK\_SEQ, BLOCK\_INTER, NUM\_STAGES, CON\_WARPGRPS≥ 2, PROD\_WARPGRPS= 1, $d_h=128, d_e \mod BLOCK\_INTER=0$ **Grid:** $x = [L/(BLOCK\_SEQ \cdot CON\_WARPGRPS)], y = H, z = B$ 1: Allocate stage/ring buffers in SRAM for Q (per consumer), R (per consumer), and K, U, V (per stage). 2: Warmup (producer): prefetch the Q tiles for all consumers in this x-block; prefetch R for subnet e=0; prefetch the first NUM\_STAGES (K, U, V) inter-tiles. 3: Producer loop (over inter-tiles): 4: for inter\_tile = NUM\_STAGES, ..., $E \cdot (d_e/\text{BLOCK_INTER}) - 1$ do wait for consumers to finish the target stage; then prefetch the next (K, U, V) tiles into that stage. 6: **if** inter\_tile is the first tile of a new subnet e **then** 7: prefetch router R rows for all consumers (current x-block). 8: end if 9: end for 10: Each consumer warpgroup $c \in \{0, \dots, CON\_WARPGRPS-1\}$ (independent, identical): 11: load its Q tile; set $O_{\text{acc}} \leftarrow 0$ . 12: for inter\_tile = $0, \dots, E \cdot (d_e/\text{BLOCK\_INTER}) - 1$ do 13: wait until producer has filled the current stage with (K, U, V); if this tile starts a new subnet, wait for $M \leftarrow Q_{\text{blk}} K_{\text{tile}}^{\top}; \quad N \leftarrow Q_{\text{blk}} U_{\text{tile}}^{\top}$ 14: $S \leftarrow \mathrm{SiLU}(M) \odot N; \quad S \leftarrow S \odot r$ 15: $\triangleright$ apply router row r $O_{\rm acc} \leftarrow O_{\rm acc} + S \, V_{\rm tile}$ 16: 17: signal producer that this stage can be reused. 18: **end for** 19: store $O_{\rm acc}$ to global output.

### Algorithm 5 SRAMFFN-BACKWARD-TK (HOPPER)

```
Require: saved Q, K, U, V, R; upstream dO
```

Meta: BLOCK\_SEQ=BLOCK\_INTER, NUM\_STAGES= 2, two consumer warpgroups, one producer **Grid:**  $x = [(E \cdot d_e)/(2 \cdot \text{BLOCK\_INTER})], y = H, z = B$ 

- 1: Assign each consumer a distinct inter-tile (A/B). Allocate per-stage SRAM for Q, dO, R; per-consumer SRAM for its (K, U, V) tile; small scratch for partial dQ/dR exchange.
- 2: Warmup (producer): prefetch (K, U, V) for both consumers' inter-tiles; prefetch first-stage Q, dO, and subnet-e router R.

### 3: Producer loop (over sequence tiles):

- 4: **for** each sequence tile t **do**
- wait for consumers to release stage  $t \mod 2$ ; prefetch Q[b, h, t], dO[b, h, t], and R[b, h, e, t] for that stage.
- 6: end for

### 7: Consumer warpgroup #0 (inter-tile A):

- 8: init  $dK_{\rm acc}$ ,  $dU_{\rm acc}$ ,  $dV_{\rm acc} \leftarrow 0$ .
- 9: **for** each sequence tile *t* **do**
- wait for producer to provide Q, dO, R for stage  $t \mod 2$ ; use preloaded (K, U, V) for A. 10:
- $M \leftarrow Q_{\text{blk}} K_{\text{A}}^{\top}; \quad N \leftarrow Q_{\text{blk}} U_{\text{A}}^{\top}$ 11:
- $A \leftarrow \text{SiLU}(M); \quad dA \leftarrow dO_{\text{blk}} V_{\text{A}}$ 12:
- $A' \leftarrow A + \operatorname{sigmoid}(M) \cdot (1-A)$ 13:
- 14:  $dR_{\text{rows}} += \text{row\_sum}(dA \odot A \odot N)$
- $dN \leftarrow (dA \odot A) \odot r; \quad dM \leftarrow (dA \odot (N \odot r)) \odot A'$ 15:
- 16:
- $\begin{array}{l} dQ^{(A)} \leftarrow dN\,U_{\rm A} + dM\,K_{\rm A} \\ dK_{\rm acc} += dM^{\top}\,Q_{\rm blk}; \quad dU_{\rm acc} += dN_{\perp}^{\top}\,Q_{\rm blk} \end{array}$ 17:
- $A_{\text{gated}} \leftarrow A \odot (N \odot r); \quad dV_{\text{acc}} += A_{\text{gated}}^{\top} dO_{\text{blk}}$ 18:
- write  $dQ^{(A)}$  and  $dR_{\text{rows}}$  to a shared slot; notify peer. 19:
- 21: store-add  $dK_{\rm acc}$ ,  $dU_{\rm acc}$ ,  $dV_{\rm acc}$  to global.

## 22: Consumer warpgroup #1 (inter-tile B):

- 23: same loop with (K, U, V) for B, producing  $dQ^{(B)}$ ,  $dR^{(B)}_{rows}$ , and its own  $dK_{acc}$ ,  $dU_{acc}$ ,  $dV_{acc}$ .
- 24: at each t: wait for peer's  $dQ^{(A)}/dR^{(A)}$ , then
- $dQ[b, h, t] += dQ^{(A)} + dQ^{(B)}, \quad dR[b, h, e, t] += dR^{(A)} + dR^{(B)},$
- 26: then free the shared slot so producer can reuse the stage.
- 27: store-add its  $dK_{\rm acc}$ ,  $dU_{\rm acc}$ ,  $dV_{\rm acc}$  to global.

### <span id="page-14-0"></span>TRAINING HYPERPARAMETERS

<span id="page-14-1"></span>All models training at particular size are trained with optimizer with hyperparameters set in Table 3:

Table 3: Training hyperparameters by model scale.

|                         | 128M            | 370M            | 1.3B            |
|-------------------------|-----------------|-----------------|-----------------|
| Learning rate           | 3e-3            | 1.5e-3          | 1e-3            |
| Learning scheduler      | cos_with_min_lr | cos_with_min_lr | cos_with_min_lr |
| Min LR                  | 1e-5            | 1e-5            | 1e-5            |
| Warmup ratio            | 0.015           | 0.015           | 0.015           |
| Adam $\bar{\beta}_1$    | 0.9             | 0.9             | 0.9             |
| Adam $\beta_2$          | 0.95            | 0.95            | 0.95            |
| Weight decay            | 1e-1            | 1e-1            | 1e-1            |
| Total batch size        | 64              | 64              | 64              |
| Segment length (tokens) | 4096            | 4096            | 4096            |
| Training steps          | 245K            | 245K            | 409K            |

D TRAINING CONFIGURATIONS

Table 4: Model and training configurations across scales and variants.

<span id="page-15-0"></span>

|                                   |          | Baseline |          | MH-FFN   | FFN      | PKV      |          | FlashMHF |          |
|-----------------------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
|                                   | 128M     | 370M     | 1.3B     | 128M     | 370M     | 370M     | 128M     | 370M     | 1.3B     |
| Params (M)                        | 117.96 M | 388.05 M | 1.323 B  | 115.70 M | 379.79 M | 386.55 M | 115.77 M | 390.08 M | 1.321 B  |
| $n_{\rm layers}$                  | 12       | 24       | 24       | 10       | 20       | 24       | 10       | 21       | 20       |
| $d_{\rm model}$                   | 892      | 1024     | 2048     | 292      | 1024     | 1024     | 292      | 1024     | 2048     |
| $n_{\rm att-heads}/d_{\rm head}$  | 12/64    | 16/64    | 32/64    | 12/64    | 16/64    | 16/64    | 12/64    | 16/64    | 32/64    |
| Intermediate size                 | 2048     | 2752     | 5504     | 2048     | 2752     | 3072     | 2048     | 2688     | 5760     |
| $n_{\rm ffn\text{-}heads}(H)/d_h$ | -/-      | -/-      | -/-      | 6/128    | 8/128    | 8/128    | 6/128    | 8/128    | 16/128   |
| Sub-network Num $E$               | ı        | 1        | ı        | 1        | ı        | ı        | 8        | 7        | 15       |
| Training steps                    | 245K     | 245K     | 409K     | 245K     | 245K     | 245K     | 245K     | 245K     | 409K     |
| Learning rate                     | 3e-3     | 1.5e-3   | 1e-3     | 3e-3     | 1.5e-3   | 1.5e-3   | 3e-3     | 1.5e-3   | 1e-3     |
| Training tokens                   | 60B      | 60B      | 100B     | 60B      | 60B      | 809      | 809      | 60B      | 100B     |
| Hidden act                        | silu     | silu     | silu     | silu     | silu     | silu     | silu     | silu     | silu     |
| initializer_range                 |          | 0.02     | 0.02     | 0.02     | 0.02     | 0.02     | 0.02     | 0.02     | 0.02     |
| max_position_embeddings           |          | 4096     | 4096     | 4096     | 4096     | 4096     | 4096     | 4096     | 4096     |
| pretraining_tp                    | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        |          |
| rms_norm_eps                      | 1e-05    | 1e-05    | 1e-05    | 1e - 05  | 1e-05    | 1e-05    | 1e-05    | 1e-05    | 1e-05    |
| rope_scaling                      | null     | null     | null     | null     | null     | null     | null     | null     | null     |
| tie_word_embeddings               | true     | false    | false    |          | false    |          |          |          | false    |
| torch_dtype                       | bfloat16 | 16       | bfloat16 | 9        | bfloat16 |          | 91       | 9        | bfloat16 |
| vocab_size                        | 50432    | 50432    | 50432    | 50432    | 50432    | 50432    | 50432    | 50432    | 50432    |
|                                   |          |          |          |          |          |          |          |          |          |

