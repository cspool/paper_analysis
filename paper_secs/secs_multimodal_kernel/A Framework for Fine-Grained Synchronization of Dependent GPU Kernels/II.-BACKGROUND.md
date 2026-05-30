# II. BACKGROUND

This section provides a background on NVIDIA GPUs and ML models.

#### A. NVIDIA Graphics Processing Units and CUDA

A parallel computation executing on NVIDIA GPUs is called a CUDA kernel. A CUDA kernel executes multiple concurrent threads organized in a 3-dimensional grid, and these threads are grouped into equally sized thread blocks. The dim3 struct in CUDA represents a 3-D grid size and identifier for both threads and thread blocks in x, y, and z dimensions. An NVIDIA GPU contains multiple Streaming Multiprocessors (SMs), each of which executes one or more thread blocks. This number of thread blocks per SM, known as occupancy, depends on the register and shared memory usage, thread block size, and number of thread blocks of the CUDA kernel.

Thread block Wave Execution Thread blocks are executed on all SMs in  $\lceil \frac{Number\_of\_TBs\_in\_Grid}{occupancy \times Number\_of\_SMs} \rceil$  waves, where the initial full waves execute occupancy  $\times$  Number\\_of\\_SMs thread blocks and the final partial wave execute the remaining thread blocks. NVIDIA has not documented the mechanism for scheduling thread blocks to SMs followed by CUDA and GPUs.

**Stream Synchronization** A CUDA *stream* is a sequence of CUDA operations that execute in the order they were issued.

```
1 //X: [B,S,H]; W_1: [H,4H/8]; W_2: [4H/8,H]
2 //1st GeMM fused with GeLU XW_1: [B,S,4*H/8]
3 XW_1 = GeLU(X × W_1)
4 //2nd GeMM XW_2: [B,S,H]
5 XW_{12} = XW_1 × W_2
```

(a) Multi-Layer Perceptron (MLP) contains two weight matrices:  $\mathbb{W}_1$  of shape  $\left[\mathbb{H}, \frac{4\mathbb{H}}{8}\right]$  and  $\mathbb{W}_2$  of shape  $\left[\frac{4\mathbb{H}}{8}, \mathbb{H}\right]$ .

```
1 //X: [B,S,H]; QKV: [H,3H/8]; W<sub>2</sub>: [H/8,H]
2 //1st GeMM XQKV: [B,S,3H/8]
3 \text{ XQKV} = \text{X} \times \text{QKV}
4 //XQ: [B,S,H/8]; XK: [B,S,H/8]; XV: [B,S,H/8]
5 XQ = XQKV[:,:,0:H/8]
                                 //1st matrix slice
6 \text{ XV} = \text{XQKV}[:,:,\text{H}/8:2*\text{H}/8]//2\text{nd} \text{ matrix slice}
7 \text{ XK} = \text{XQKV}[:,:,2*H/8:]
                                 //3rd matrix slice
8 //Cached Attention Mechanism
9 //CachedK: [H/8,S',B ]
10 //CachedV: [B ,S',H/8]
11 P = XQ × Concat (CachedK, XK.T)
12 R = Softmax(Dropout(P))
13 T = R \times Concat(CachedV, XV)
14 CachedV[:S'+S:]_=_XV
15 \text{ CachedK}[:S'+S:] = XK.T
16 //2nd GeMM XW<sub>2</sub>: [B,S,H]
17 \text{ XW}_{12} = \text{R} \times \text{W}_2
```

<span id="page-2-3"></span><span id="page-2-2"></span>(b) Attention contains two weight matrices: QKV of shape  $\left[\frac{3H}{8},H\right],$  and  $W_2$  of shape  $\left[\frac{H}{8},H\right].$  Attention caches generated keys and values for each token to avoid recomputation of all previous tokens during inference.

Fig. 2: Architecture of Multi-Layer Perceptron (MLP) and Attention of GPT-3, where H is 12288. Model parallelism on 8 GPUs divides weight matrices of both layers among 8 GPUs. Both takes an input matrix X of shape [B,S,H] and obtain the result  $XW_{12}$  of the same shape. B is the number of batched requests, S is the sequence length, H is the hidden dimension, and S' is the sum of processed and generated tokens.

When two dependent kernels are invoked on the same stream, the consumer-kernel is not started before all thread blocks of the producer-kernel have finished their execution. We call this synchronization *stream synchronization*. We can invoke independent CUDA kernels on different streams to execute kernels concurrently. A stream has an associated priority value, such that operations on a higher priority stream are issued before a lower priority stream.

