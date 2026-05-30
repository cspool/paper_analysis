# <span id="page-22-0"></span>D Balancing Key-Value Norms and Low-Rank Approximation

This appendix elaborates on the Key-Value (KV) balancing technique and the subsequent joint low-rank approximation applied to the NoPE (No Positional Encoding) components of the keys and the values, as mentioned in Section 4.3 of the main paper. After the RoRoPE procedure (Section 4.2), the key projection matrix  $W^K$  is effectively split into two components:  $W^{DK}_{\text{RoPE}} \in \mathbb{R}^{d \times D}$  corresponding to the single head that retains RoPE, and  $W^{DK}_{\text{NoPE}} \in \mathbb{R}^{(g-1)d \times D}$  corresponding to the remaining g-1 head components that do not use RoPE. The value projection matrix is denoted as  $W^{DV} \in \mathbb{R}^{gd \times D}$ .

