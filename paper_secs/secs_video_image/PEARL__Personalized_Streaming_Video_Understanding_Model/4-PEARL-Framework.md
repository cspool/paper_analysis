# 4 PEARL Framework

To address the challenges of the task of PSVU, we propose a plug-and-play framework, PEARL. As illustrated in Fig. [3,](#page-8-0) it dynamically defines concepts at specific timestamps of streaming video via user instructions and provides realtime responses to user queries in subsequent timestamps.

In Section [4.1,](#page-7-0) we present a formal formulation of the task. In Section [4.2,](#page-7-1) we propose a Dual-grained Memory System to store historical video stream clips and defined concepts. In Section [4.3,](#page-8-1) we present an efficient Concept-aware Retrieval Algorithm for fast retrieval and response.

#### <span id="page-7-0"></span>4.1 Formulation

Formally, we define a streaming video as an infinite sequence V = [X1, X2, . . . ], where X<sup>i</sup> denotes a video clip representing a semantic scene. Throughout the stream, a user can dynamically introduce new concepts at any timestamp t<sup>c</sup> via instructions, forming an evolving set of defined concepts C = {C1, C2, . . . }. For a query Q issued at time t<sup>q</sup> ≥ tc, the model M must dynamically construct a context to generate a response A:

$$A = \mathcal{M}(\mathcal{C}_{sub}, \mathcal{V}_{context}, Q) \tag{1}$$

where Csub ⊆ C is the query-relevant concept subset, and Vcontext is the necessary visual context. Solving this requires overcoming two key challenges: the prohibitive cost of maintaining unbounded stream history alongside evolving concepts, and the difficulty of accurately retrieving personalized Csub and Vcontext in real-time. This motivates our design of a scalable dual-grained memory and a concept-aware retrieval strategy.

