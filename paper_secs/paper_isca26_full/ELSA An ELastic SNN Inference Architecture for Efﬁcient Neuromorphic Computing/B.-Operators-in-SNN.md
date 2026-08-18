# B. Operators in SNN

1) Matrix Multiplication (MM): Unlike conventional MM with two continuous-valued operands, SNNs use spike-continuous MM (MM-sc) and spike-spike MM (MM-ss). Spiking convolution and linear layers are implemented with MM-sc, while spiking attention uses MM-ss. Following SpikeZIP-TF [4], we realize MM-ss with two MM-sc operators by treating spike tracers as continuous operands.

2) Miscellaneous Operators: Although MM operators dominate execution time and energy, correct SNN inference also requires the miscellaneous operators summarized in Tab. I. We follow SpikeZIP-TF [4] for spiking softmax (ssoftmax) and spiking layer normalization (slayernorm), and implement image-to-column transformation and residual addition as router-side broadcasts.

#### III. MOTIVATIONS

