# C.1.1 Illustrative Example of FreqFold

Let's consider a scenario with g=2 key heads, and each head has  $d_{head}=8$  dimensions. Thus, there are d/2=8/2=4 distinct RoPE frequency indices per head, which we denote as  $\phi_1,\phi_2,\phi_3,\phi_4$ . The total number of dimensions is  $2\times 8=16$ . The RoPE angles for these 16 dimensions could be conceptualized as follows (repeating for each pair, and across heads):

- Head 1 (dims 1-8):  $(\phi_1, \phi_1), (\phi_2, \phi_2), (\phi_3, \phi_3), (\phi_4, \phi_4)$
- Head 2 (dims 9-16):  $(\phi_1, \phi_1), (\phi_2, \phi_2), (\phi_3, \phi_3), (\phi_4, \phi_4)$

Case 1: RoRoPE without FreqFold For each frequency index  $\phi_l$ , RoRoPE groups the corresponding dimensions from all g=2 heads. Each such group forms  $2g=2\times 2=4$ -dimensional vectors (across N samples).

- Group for  $\phi_1$ : Dimensions  $\{1,2\}$  from Head 1 and  $\{9,10\}$  from Head 2. PCA is applied to these N samples of 4D vectors.
- Group for  $\phi_2$ : Dimensions  $\{3,4\}$  from Head 1 and  $\{11,12\}$  from Head 2. PCA is applied to these N samples of 4D vectors.
- Group for  $\phi_3$ : Dimensions  $\{5,6\}$  from Head 1 and  $\{13,14\}$  from Head 2. PCA is applied to these N samples of 4D vectors.
- Group for  $\phi_4$ : Dimensions  $\{7,8\}$  from Head 1 and  $\{15,16\}$  from Head 2. PCA is applied to these N samples of 4D vectors.

Here, RoRoPE performs 4 separate PCA operations.

Case 2: RoRoPE with 2D-FreqFold 2D-FreqFold implies we are pairing up original frequencies. Suppose FreqFold approximates  $\phi_1 \approx \phi_2$  (calling this effective frequency  $\Phi_A = \phi_1$ ) and  $\phi_3 \approx \phi_4$  (calling this  $\Phi_B = \phi_3$ ).

- Effective Group for Φ<sub>A</sub>: This group now includes all dimensions originally associated with φ<sub>1</sub> OR φ<sub>2</sub>.
  - Original  $\phi_1$ -dimensions:  $\{1,2\}$  from Head 1;  $\{9,10\}$  from Head 2. (Forms a 4D segment  $S_{\phi_1}$ )
  - Original  $\phi_2$ -dimensions:  $\{3,4\}$  from Head 1;  $\{11,12\}$  from Head 2. (Forms a 4D segment  $S_{\phi_2}$ )

With FreqFold, these segments  $S_{\phi_1}$  and  $S_{\phi_2}$  are concatenated. PCA is now applied to the N samples of (4+4)=8-dimensional vectors formed by  $[S_{\phi_1},S_{\phi_2}]$ . Effectively, dimensions  $\{1,2,3,4\}$  from Head 1 are combined with  $\{9,10,11,12\}$  from Head 2.

- Effective Group for  $\Phi_B$ : Similarly, this group includes dimensions originally for  $\phi_3$  OR  $\phi_4$ .
  - Original  $\phi_3$ -dimensions:  $\{5,6\}$  from Head 1;  $\{13,14\}$  from Head 2. (Forms  $S_{\phi_3}$ )
  - Original  $\phi_4$ -dimensions:  $\{7,8\}$  from Head 1;  $\{15,16\}$  from Head 2. (Forms  $S_{\phi_4}$ )

PCA is applied to the N samples of 8-dimensional vectors formed by  $[S_{\phi_3}, S_{\phi_4}]$ .

Here, RoRoPE with FreqFold performs 2 PCA operations, but each operates on larger, 8-dimensional vectors which are concatenations of what were previously separate PCA targets.

