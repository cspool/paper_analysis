# <span id="page-3-1"></span>4 Offline Refactoring Engine

The Refactoring Engine is the offline part that methodically transforms a standard, pre-trained MoE model into a fine-grained, elastic artifact.

### 4.1 Neuron Activation Profiler

The Neuron Activation Profiler is the first component in the MoE-Prism refactoring engine. Its purpose is to create a detailed functional fingerprint of each expert by capturing its runtime behavior on representative data. To achieve this, the Profiler processes a calibration dataset through the pretrained model and intercepts the intermediate activations within each expert's FFN layer. In modern LLMs, these FFNs are typically SwiGLU layers, whose structure allows for clean decomposition:

$$FFN(X) = (SiLU(X \cdot W_{gate}) \odot (X \cdot W_{up})) \cdot W_{down} \quad (1)$$

The key insight enabling our approach is that the computation for each column of the intermediate activation matrix,  $\mathbf{A} = \mathrm{SiLU}(X \cdot W_{\mathrm{gate}}) \odot (X \cdot W_{\mathrm{up}})$ , is independent. This allows us to define a "neuron" as the collection of weights responsible for a single column of  $\mathbf{A}$  and its corresponding contribution to the output (the j-th columns of  $W_{\mathrm{gate}}/W_{\mathrm{up}}$  and j-th row of  $W_{\mathrm{down}}$ ). The result of this stage is a set of activation matrices  $\{\mathbf{M}_e\}$ , one for each expert e. Each matrix  $\mathbf{M}_e \in \mathbb{R}^{B \times C}$  (for B tokens and C neurons) serves as a detailed profile of the expert's behavior and is the primary input for the subsequent partitioning stage.

