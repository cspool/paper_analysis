# E. Impact of Rotation on Different Models

Fig. [6](#page-15-0) showcases the effects of (random Walsh-Hadamard) rotation applied to several exemplary layers in Pythia and Llama models, and demonstrates that sometimes applying the rotation can lead to undesirable results where new outlier values emerge. Fig. [7](#page-16-0) presents a comprehensive view of the effects of applying rotations to the weights, activation, and KV cache of different layers. Notice that the RoSTE algorithm only applies rotation when a reduction of quantization error is observed in the respective layers. Moreover, from the figure we observe that in general, the last layers of Pythia model do not benefit from applying rotation, while the rotation effects on Llama model are generally beneficial.

<span id="page-15-0"></span>![](_page_15_Figure_5.jpeg)

Figure 6. Visualizations of Input Activations in Pythia and Llama Models before and after rotation.

☹️ We conjecture that several architectural differences between Pythia and Llama contribute to this discrepancy. First, Pythia does not utilize Gated Linear Units (GLU) in its MLP layers, a feature that is integral to Llama. Second, Pythia employs layer normalization (LayerNorm) instead of root mean square normalization (RMSNorm) which is used in Llama. Finally, Pythia adopts a parallel residual connection for attention and feed-forward layers, in contrast to the sequential residual connection found in Llama.

