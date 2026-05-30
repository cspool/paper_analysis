# J. Adding Uniform Noise to weights

Quantization noise and uniform weight noise share similarities in that both introduce perturbations to the model's parameters. However, quantization noise specifically arises from the discretization of the weights, which can lead to a more structured form of regularization due to the rounding or truncation during the quantization process. In contrast, uniform weight noise typically adds random perturbations with a uniform distribution, which may not exhibit the same structured regularization properties.

In Table [15,](#page-17-0) we provide the results of our ablation study on the PACS dataset with uniform noise with different minimum and maximum value:

<span id="page-17-0"></span>

| Noise                        | OOD Accuracy |
|------------------------------|--------------|
| No noise                     | 84.7 ± 0.5   |
| Uniform(-0.0001, 0.0001)     | 82.9 ± 0.6   |
| Uniform(-0.00005, 0.00005)   | 83.8 ± 0.5   |
| Uniform(-0.00001, 0.00001)   | 85.1 ± 0.4   |
| Uniform(-0.000005, 0.000005) | 85.6 ± 0.3   |

Table 15. OOD Accuracy under different noise levels

