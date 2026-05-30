# <span id="page-11-0"></span>B. Algorithm Details

```
Algorithm 1 Constructing re-index vector
  Input: Routing choice R with shape (N,).
  Initialize: Tensor ctr with shape (E,) initialized with 0.
  Parallel for i = 0 to N − 1 do
     atomicAdd(ctr[R[i]], 1)
  end for
  Parallel for i = 0 to E − 1 do
     ctr[i] = BLK · ⌊ctr[i] / BLK⌋
  end for
       N′ =
             E
              P−1
              i=0
                  ctr[i]
  Initialize: Tensor v with shape (N′
                                      ,) initialized with -1, and tensor idx with shape (1 + E,) initialized with 0.
  Parallel for i = 0 to E − 1 do
     ctr[i] = BLK · ⌊ctr[i] / BLK⌋
  end for
  for i = 1 to E do
     idx[i] = idx[i − 1] + ctr[i − 1]
  end for
  Copy: idx = idx
  Parallel for i = 0 to N − 1 do
     pos = atomicAdd(idx[R[i]], 1)
     v[pos] = i
  end for
  Output: Tensor v and idx
```

### Algorithm 2 Expert-specific matrix multiplication

```
Input: Routing choice R with shape (N,), vector v with length N′
                                                                  , input tokens x with shape (N, D1), weights w with
shape (E, D1, D2) and bias b with shape (E, D2)
Initialize: Output tokens y with shape (N, D2) initialized with 0.
Parallel for i in range (0, N′
                             , BLK) do
  Parallel for j in range (0, D2, BLK) do
     exp = R[v[i]]
    Initialize zero tensor c with shape (BLK, BLK)
    load bsub = b[exp, j : j + BLK]
     c = bsub.repeat(BLK, 1)
    for k in range (0, D1, BLK) do
       Initialize zero tensor xsub with shape (BLK, BLK)
       Parallel for t = 0 to BLK do
         load xsub[t] = x[v[i + t], k : k + BLK]
       end for
       load wsub = w[exp, k : k + BLK, j : j + BLK]
        c = c + xsub · wsub
    end for
    Parallel for t = 0 to BLK do
       Write back: y[v[i + t], j : j + BLK] = c[t]
    end for
  end for
end for
Output: Tensor y
```

#### Algorithm 3 Expert-specific summation

```
Input: Routing choice \mathcal{R} with shape (N, ), vector v with length N', vector idx with length 1 + E and input tokens x
with shape (N, D)
Initialize: Output tokens y with shape (E, D) initialized with 0.
Parallel for i = 0 to E - 1 do
  Parallel for j in range (0, D, BLK) do
      exp = \mathcal{R}[v[idx[i]]]
     Initialize zero tensor c with shape (1, BLK)
     Initialize zero tensor x_{sub} with shape (BLK, BLK)
     for k in range (idx[i],idx[i+1],\operatorname{BLK}) do
       Parallel for t=0 to BLK do
          load x_{sub}[t] = x[v[k+t], j:j+BLK]
       end for c = c + \sum_{t=0}^{\mathtt{BLK}-1} x_{sub}[t]
  Write back: y[exp, j: j + BLK] = c
  end for
end for
Output: Tensor y
```

#### **Algorithm 4** Expert-specific transposed matrix multiplication

```
Input: Routing choice \mathcal{R} with shape (N,), vector v with length N', vector idx with length 1+E, the first token batch
x_1 with shape (N, D_1) and the second token batch x_2 with shape (N, D_2)
Initialize: Output y with shape (E, D_1, D_2) initialized with 0.
Parallel for i = 0 to E - 1 do
   Parallel for m in range (0, D_1, BLK) do
      Parallel for n in range (0, D_2, BLK) do
           exp = \mathcal{R}[v[idx[i]]]
          Initialize zero tensor c with shape (BLK, BLK)
          \begin{array}{l} \textbf{Initialize} \ \text{zero tensor} \ x_{sub}^1 \ \text{with shape} \ (\texttt{BLK}, \texttt{BLK}) \\ \textbf{Initialize} \ \text{zero tensor} \ x_{sub}^2 \ \text{with shape} \ (\texttt{BLK}, \texttt{BLK}) \end{array}
          for k in range (idx[i], idx[i+1], BLK) do
             Parallel for t=0 to BLK do
                  \begin{array}{l} \mathbf{load} \ x_{sub}^1[t] = x_1[v[k+t], m:m + \mathtt{BLK}] \\ \mathbf{load} \ x_{sub}^2[t] = x_2[v[k+t], n:n + \mathtt{BLK}] \end{array} 
              c = c + x_{sub}^1.transpose() \cdot x_{sub}^2
          end for
          Write back in parallel:
               y[exp, m: m + BLK, n: n + BLK] = c
      end for
   end for
end for
Output: Tensor y
```

We present the algorithm details for re-index vector construction as well as the expert-specific operators, and taking top-1 routing as an example for illustration, shown in Algorithm 1, 2, 3 and 4. In Algorithm 1, we re-arrange the routing choice vector  $\mathcal{R}$  into re-indexed token vector v, along with the token index starting vector idx, which satisfies idx[0] = 0 and idx[E] = N'. We provide the length and range for the vectors in Table 4.

In Algorithm 2, 3 and 4, we assume that the feature dimension for each tensor is all divisible by BLK. Notice that when

Table 4. Explanation for the auxiliary vectors.

<span id="page-13-1"></span>

| vector         | length   | range                                                   |
|----------------|----------|---------------------------------------------------------|
| $\mathcal{R}$  | $\mid N$ | $0 \le \mathcal{R}[i] < E \text{ for } 0 \le i < N$     |
| $\overline{v}$ | N'       | $\big  \qquad 0 \le v[i] < N \text{ for } 0 \le i < N'$ |
| idx            | 1+E      | $0 \le idx[i] < N' \text{ for } 0 \le i < 1 + E$        |

accessing the re-index vector v, we may get value -1, since the workload for each expert is dynamic. In this case, we would skip this index, and remain zero for the temporary loading variable.

We also provide the formulation for top-k routing with our expert-specific operators, and take a comparison with top-1 routing. We denote the routing choice for top-k as  $\{\mathcal{R}_i(\boldsymbol{x})\}_{i=0}^{k-1}$ , and other symbols keep consistent with Figure 2. We present the comparison on formulations for MoE forward and backward propagation in Table 5.

<span id="page-13-2"></span> $\it Table~5.$  Comparison between top-1 and top- $\it k$  routing for the formulation of MoE forward and backward with our expert-specific operators.

| Stage    | Notation Layer |            | Expert-Specific Formulation (top-1 routing)                                                                                                                             | Expert-Specific Formulation (top-k routing)                                                                                                                                                                                                                                             |  |  |  |  |
|----------|----------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|--|
|          | 1              | 1st MLP    | $\boldsymbol{y}_1 = \textit{ESMM}(\boldsymbol{x}, \boldsymbol{W}_1, \boldsymbol{b}_1, \mathcal{R}(\boldsymbol{x}))$                                                     | $\bigg  \{\boldsymbol{y}_1^i\}_{i=0}^{k-1}: \boldsymbol{y}_1^i = \textit{ESMM}(\boldsymbol{x}, \boldsymbol{W}_1, \boldsymbol{b}_1, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                                      |  |  |  |  |
| Forward  | 2              | Activation | $\boldsymbol{y}_2 = \mathcal{F}(\boldsymbol{y}_1)$                                                                                                                      | $\big \{\boldsymbol{y}_2^i\}_{i=0}^{k-1}:\boldsymbol{y}_2^i=\mathcal{F}(\boldsymbol{y}_1^i)$                                                                                                                                                                                            |  |  |  |  |
|          | 3              | 2nd MLP    | $\boldsymbol{y} = \textit{ESMM}(\boldsymbol{y}_2, \boldsymbol{W}_2, \boldsymbol{b}_2, \mathcal{R}(\boldsymbol{x}))$                                                     | $\boxed{\boldsymbol{y} = \sum_{i=0}^{k-1} \textit{ESMM}(\boldsymbol{y}_2^i, \boldsymbol{W}_2, \boldsymbol{b}_2, \mathcal{R}_i(\boldsymbol{x}))}$                                                                                                                                        |  |  |  |  |
|          | 4              |            | $\frac{\partial \ell}{\partial \boldsymbol{b}_2} = \textit{ESS}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \mathcal{R}(\boldsymbol{x}))$                            | $\frac{\partial \ell}{\partial \boldsymbol{b}_2} = \sum_{i=0}^{k-1} \textit{ESS}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                                         |  |  |  |  |
|          | (5)            | 2nd MLP    | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1}, \mathcal{R}(\boldsymbol{x}))$        | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \sum_{i=0}^{k-1} \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                   |  |  |  |  |
| Backward | 6              |            | $\frac{\partial \ell}{\partial \boldsymbol{y}_2} = \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \boldsymbol{W}_2^T, null, \mathcal{R}(\boldsymbol{x}))$ | $   \left\{ \frac{\partial \ell}{\partial \boldsymbol{y}_{2}^{i}} \right\}_{i=0}^{k-1} : \frac{\partial \ell}{\partial \boldsymbol{y}_{2}^{i}} = \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}}, \boldsymbol{W}_{2}^{T}, \textit{null}, \mathcal{R}_{i}(\boldsymbol{x})) $ |  |  |  |  |
| Buckward | 7              | Activation | $\frac{\partial \ell}{\partial \boldsymbol{y}_1} = \frac{\partial \ell}{\partial \boldsymbol{y}_2} \odot \mathcal{F}'(\boldsymbol{y}_1)$                                | $\left\{\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}\right\}_{i=0}^{k-1}:\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}=\frac{\partial \ell}{\partial \boldsymbol{y}_2^i}\odot\mathcal{F}'(\boldsymbol{y}_1^i)$                                                                 |  |  |  |  |
|          | 8              |            | $\frac{\partial \ell}{\bm{b}_1} = \textit{ESS}(\frac{\partial \ell}{\partial \bm{y}_1}, \mathcal{R}(\bm{x}))$                                                           | $\frac{\partial \ell}{\boldsymbol{b}_1} = \sum_{i=0}^{k-1} \textit{ESS}(\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                                              |  |  |  |  |
|          | 9              | 1st MLP    | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1}, \mathcal{R}(\boldsymbol{x}))$        | $\frac{\partial \ell}{\partial \boldsymbol{W}_1} = \sum_{i=0}^{k-1} \textit{ESTMM}(\boldsymbol{x}, \frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \mathcal{R}_i(\boldsymbol{x}))$                                                                                                   |  |  |  |  |
|          | 10             |            | $\frac{\partial \ell}{\partial \boldsymbol{x}} = \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}_1}, \boldsymbol{W}_1^T, null, \mathcal{R}(\boldsymbol{x}))$ | $ \frac{\partial \ell}{\partial \boldsymbol{x}} = \sum_{i=0}^{k-1} \textit{ESMM}(\frac{\partial \ell}{\partial \boldsymbol{y}_1^i}, \boldsymbol{W}_1^T, \textit{null}, \mathcal{R}_i(\boldsymbol{x})) $                                                                                 |  |  |  |  |

### <span id="page-13-0"></span>C. Experimental Details

We provide the details of our CUDA program via enumerating the shape of the thread block and thread grid for expert-specific operators in a single MoE layer in Table 6.

We also provide the PyTorch-style pseudocode for the proxy task we used to examine the computing capacity of the heterogeneous devices, as shown in Algorithm 5. We adopt a for loop composed of large matrix multiplications with the same scale as the test program.

<span id="page-14-0"></span>Table 6. Shape of the thread block and thread grid for expert-specific operators in one MoE layer. We take top-1 routing as an example, where N' denotes the length of the re-index vector, which is slightly larger than N and is divisible by BLK. Thread blocks are all defined with the same shape to facilitate the fused kernel.

|          | Operator       | Input                                                                                                                                                                | Output                                                  | Thread Block     | Thread Grid                                                                                                                                                                                |  |  |
|----------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|
| forward  | <br>  ESMM<br> | $\begin{array}{c c c} & \boldsymbol{x} & (N,D_1) \\ \hline & \boldsymbol{w} & (E,D_1,D_2) \\ \hline & \boldsymbol{b} & (E,D_2) \\ \hline \end{array} \boldsymbol{y}$ | $  (N, D_2)  $                                          | (WARP,<br>TIMES) | $\big( \lceil N' / \texttt{BLK} \rceil, \ \lceil D_2 / (\texttt{TIMES} \cdot \texttt{BLK}) \rceil \big)$                                                                                   |  |  |
|          | ESMM           | $\begin{array}{ c c c c c c c c c c c c c c c c c c c$                                                                                                               | $(N, D_1)$                                              | (WARP,<br>TIMES) | $\left[ \begin{array}{c} \left( \lceil N' / \mathtt{BLK} \rceil, \\ \lceil D_1 / (\mathtt{TIMES} \cdot \mathtt{BLK}) \rceil \right) \end{array} \right]$                                   |  |  |
| vard     | ESS            | $  x   (N, D_2)   y$                                                                                                                                                 | $(E,D_2)$                                               | (WARP,<br>TIMES) | $(E, \lceil D_2/(\mathtt{TIMES} \cdot \mathtt{BLK}) \rceil)$                                                                                                                               |  |  |
| backward | ESTMM          | $\begin{array}{ c c c c c c c c c c c c c c c c c c c$                                                                                                               | $\left \begin{array}{c} (E,D_1,D_2) \end{array}\right $ | (WARP,<br>TIMES) | $D_1/(\text{TIMES} \cdot \text{BLK}),$                                                                                                                                                     |  |  |
|          | <br>  ESFK     | $ \begin{array}{c c c c c c c c c c c c c c c c c c c $                                                                                                              | $   (N, D_1) $ $   (E, D_2) $ $   (E, D_1, D_2) $       | (WARP,           | $(E, \lceil D_1/(\texttt{TIMES} \cdot \texttt{BLK}) \rceil, \\ \lceil N'/\texttt{BLK} \rceil + \lceil D_2/\texttt{BLK} \rceil + \\ \lceil D_2/(\texttt{TIMES} \cdot \texttt{BLK}) \rceil)$ |  |  |

#### Algorithm 5 PyTorch-style pseudocode of MoE pipeline.

```
import torch\nimport time

device = 'cuda'
size = 2048
times = 1024

start_time = time.time()
for j in range(times):
    mat1 = torch.randn(size, size, device=device)
    mat2 = torch.randn(size, size, device=device)
    y = torch.matmul(mat1, mat2)\nend_time = time.time()

print(end_time - start_time)
```

