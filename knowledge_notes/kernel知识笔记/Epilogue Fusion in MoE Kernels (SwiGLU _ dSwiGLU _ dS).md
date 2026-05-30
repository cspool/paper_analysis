## Epilogue Fusion in MoE Kernels (SwiGLU / dSwiGLU / dS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Epilogue Fusion 是将 GEMM 输出后处理与多个原本需单独 kernel launch 的操作融合。SonicMoE 实现：(1) forward up-proj A kernel：WGMMA → SwiGLU activation + TMA store 融合在 epilogue；(2) backward down-proj dH kernel：WGMMA → dA = Broadcast(s)·dA' + dSwiGLU(dA, H) + dS = ⟨dA',A⟩ + A' = Broadcast(s)·A + TMA store。4 个操作融合于同一 kernel epilogue，消除了额外 kernel launch 和输出/输入 HBM 往返。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
dH kernel epilogue（per tile, per expert）：

```
// 输入: dA'_e = dO_e W_{2,e}^T (WGMMA accumulator), s_e, H_e

// 1. dA + SwiGLU recompute + dSwiGLU
for t in tile:
    dA[t,:] = s[t] * dA'[t,:]
    gate = sigmoid(H[t,:n])
    A[t,:n] = gate * H[t,n:2n]
    dH[t,:n] = dA[t,:n] * A[t,:n] * gate * (1-gate)
    dH[t,n:2n] = gate * dA[t,:n]

// 2. dS = sum over n of dA'[t,i] * A[t,i]
dS[t] = dot_product(dA'[t,:], A[t,:])

// 3. A' preparation for dW2
A'[t,:] = s[t] * A[t,:]
// TMA store: dH, dS, A' → HBM
```

SonicMoE 选择 dS=⟨dA', A⟩（vs ScatterMoE 的 dS=⟨dO, Y⟩）：(1) 0 额外 HBM 访问；(2) reduce over n vs d：节省 log₂(d/n) 轮 reduction；(3) 不需缓存 Y（节省 2TKd bytes activation）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CuTe-DSL 中通过自定义 epilogue functor 实现。dH kernel 依赖 Ping-Pong scheduling 在 hopper 上维持 high TFLOPS despite heavy epilogue。Blackwell 上通过 TMEM 2-stage + UMMA 实现更好 overlap。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
