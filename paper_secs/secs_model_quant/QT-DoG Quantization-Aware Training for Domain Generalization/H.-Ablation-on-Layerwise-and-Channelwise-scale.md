# H. Ablation on Layerwise and Channelwise scale

We conducted an ablation study where we set s at the layer level, rather than on a per-channel basis. We see that Channelwise s can lead to 1.5% accuracy as compared to layerwise s. The results of this experiment on the PACS dataset with 7 bit quantization are shown below:

| Scale           | OOD Accuracy |
|-----------------|--------------|
| No quantization | 84.7 ± 0.5   |
| Layerwise       | 86.3 ± 0.4   |
| Channelwise     | 87.8 ± 0.3   |

Table 13. OOD Accuracy with channelwise vs layerwise Scaling factor for quantization.

