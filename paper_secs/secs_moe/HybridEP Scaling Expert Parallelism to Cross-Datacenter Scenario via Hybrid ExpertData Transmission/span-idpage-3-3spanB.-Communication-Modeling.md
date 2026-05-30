# <span id="page-3-3"></span>B. Communication Modeling

All-to-All Communication Modeling. Figure 5(a) shows A2A communication details [50]. Specifically, given G GPUs, the data D on each GPU will be split into G chunks of size  $\frac{D}{G}$ . Then, G-1 chunks will be sent to other GPUs through A2A, while 1 chunk remains on the local GPU. Therefore, for GPU set  $G^{A2A}$  that participates in A2A communication, the overall traffic is expressed as:

$$V^{A2A} = \frac{D}{|G^{A2A}|} * (|G^{A2A}| - 1), Lat_{comm}^{A2A} = \frac{V^{A2A}}{B}, \quad (3)$$

where B is bandwidth. While D, B are constants,  $\underline{Lat_{comm}^{A2A}}$  remains almost constant with increased  $|G^{A2A}|$ .

All-Gather Communication Modeling. Figure 5(b) shows AG communication details. Specifically, the expert parameters (1)  $P_E$  on each GPU will be sent to other G-1 GPUs through

AG. Therefore, for GPU set  $G^{AG}$  that participates in AG, communication traffic is expressed as:

$$V^{AG} = P_E * (|G^{AG}| - 1), Lat_{comm}^{AG} = \frac{V^{AG}}{R}.$$
 (4)

Therefore,  $Lat_{comm}^{AG}$  increases linearly with  $|G^{AG}|$ .

Relationships between Two Communications. A2A can be seamlessly transformed into AG. If an expert has been obtained through AG, then the corresponding data chunk is not necessary to be transmitted through A2A. When the *i*-th GPU  $G_i$  uses AG to collect expert  $P_E$  from  $G_j$ , the A2A's traffic changes from  $D*\frac{G-1}{G}$  to  $D*\frac{G-1}{G}-\frac{D}{G}$ , while the AG's traffic changes from 0 to  $P_E$ . Therefore, when A2A's traffic decreases by  $\frac{D}{G}$ , AG's traffic increases by  $P_E$ .

**Communication Stream Modeling.** Its latency comes from both A2A and AG, which can be expressed as:

$$Lat_{comm} = Lat_{comm}^{AG} + 2Lat_{comm}^{A2A}.$$
 (5)

where A2A performs twice before and after expert computation, and AG only performs once as experts do not need to be sent back to their original GPUs.

