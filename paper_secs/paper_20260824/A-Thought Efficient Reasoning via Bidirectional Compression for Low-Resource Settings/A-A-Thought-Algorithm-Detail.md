# A A\*-Thought Algorithm Detail

Algorithm 1 shows the details of the A\* search algorithm to compress lengthy CoTs.

#### <span id="page-13-0"></span>Algorithm 1 A\*-Thought algorithm for compressing lengthy CoTs

```
1: Input:
            q: question; t: original CoT; t<sub>sort</sub>: thought list sorted by BIS; s: solution; V: verification
       model; k_{\min}: min verification depth; k_{\max}: max search depth; W: number of observable nodes
            \mathbf{t'} \subseteq \mathbf{t}: compressed thought path
  5: procedure A* SEARCH
                                                                                                                                                   ▷ (1) Initialization
               \mathcal{Q} \leftarrow \mathbf{t}_{\mathrm{sort}}
  6:
              \mathbf{t}_{k}' \leftarrow \mathcal{Q}.pop()
  7:
              k = 1
  8:
              while Q not empty do
  9:
                     10:
11:
12:
                     end if
                                                                                                                                                     ⊳ (2) Verification
                      \label{eq:linear_continuity}  \mbox{if } k \geq k_{\min} \mbox{ and } \mathcal{V}\left(\mathbf{q}+\mathbf{t}_k'\right) == \mathbf{s} \mbox{ then} \\  \mbox{return } \mathbf{t}' = \mathbf{t}_k' 
13:
14:
15:
                                                                                                                                                     ▷ (3) Exploration
                     \{\mathbf{r}_1,\ldots,\mathbf{r}_W\}\leftarrow \text{first }W \text{ elements in }\mathcal{Q}
16:
                     for \mathbf{r}_w in \{\mathbf{r}_1, \dots, \mathbf{r}_W\} do
f(\mathbf{t}_k' + \mathbf{r}_w) = g(\mathbf{t}_k' + \mathbf{r}_w) + h(\mathbf{t}_k' + \mathbf{r}_w)
17:
18:
19:
                     end for
                     \hat{\mathbf{r}}_{w} = \operatorname{argmin}_{w \in \{1, \cdots, W\}} f\left(\mathbf{t}'_{k} + \mathbf{r}_{w}\right)
20:
                     \mathbf{t}_{k+1}' = \langle \mathbf{t}_k', \hat{\mathbf{r}}_w \rangle
Q.pop(\hat{\mathbf{r}}_w)
21:
22:
                     k = k + 1
23:
              end while
24:
              return \mathbf{t}'_k
25:
26: end procedure
```

