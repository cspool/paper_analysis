# 3 Method

In this section, we introduce our proposed BI-DiffSR. First, we describe the structural designs suitable for binarization: *consistent-pixel-downsample* (CP-Down), *consistent-pixel-upsample* (CP-Up), and *channel-shuffle-fusion module* (CS-Fusion). The CP-Down and CP-Up achieve dimension adjustment and ensure the transfer of full-precision information. The CS-Fusion effectively integrates different features within the skip connection. Secondly, we present the dynamic designs tailored for varying activations: *timestep-aware redistribution* (TaR) and *activation function* (TaA). The TaR and TaA enhance the representational learning of the binarized modules across multiple timesteps.

