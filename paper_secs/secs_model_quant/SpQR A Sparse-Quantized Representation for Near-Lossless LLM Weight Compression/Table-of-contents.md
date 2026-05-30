# Table of contents

| 1 | Introduction                                                                                                                                             | 1           |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| 2 | Related Work                                                                                                                                             | 3           |
| 3 | Quantization sensitivity of LLM weights<br>3.1<br>Parameter sensitivity under quantization<br><br>3.2<br>Exploring parameter sensitivity                 | 4<br>4<br>4 |
| 4 | SpQR: A Sensitivity-aware compressed representation<br>4.1<br>Overview<br><br>4.2<br>Implementing and Leveraging the Sparse Quantized Representation<br> | 5<br>5<br>7 |
| 5 | Experimental Validation                                                                                                                                  | 8           |
| 6 | Discussion & Limitations                                                                                                                                 | 11          |
| 7 | Acknowledgements                                                                                                                                         | 11          |
| A | Additional weight sensitivity analysis                                                                                                                   | 15          |
| B | Experimental Configurations                                                                                                                              | 18          |
| C | Hyperparameter sensitivity                                                                                                                               | 18          |
| D | Estimating model size                                                                                                                                    | 18          |
| E | Choice of optimal configuration for fixed average number of bits                                                                                         | 19          |
| F | Additional results for near-lossless compression                                                                                                         | 20          |
| G | Choice of optimal LLM configuration for specific hardware                                                                                                | 20          |
| H | Sensitivity to random seed                                                                                                                               | 22          |
| I | Generative examples                                                                                                                                      | 22          |
| J | Broader impact                                                                                                                                           | 22          |
| K | On the use of LLMs in this work                                                                                                                          | 26          |

