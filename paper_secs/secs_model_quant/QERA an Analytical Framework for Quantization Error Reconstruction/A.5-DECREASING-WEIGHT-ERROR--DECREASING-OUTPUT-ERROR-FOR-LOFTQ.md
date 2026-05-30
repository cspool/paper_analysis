# A.5 DECREASING WEIGHT ERROR ̸= DECREASING OUTPUT ERROR FOR LOFTQ

We provide the weight approximation error, ||<sup>W</sup> <sup>−</sup> <sup>W</sup><sup>f</sup> <sup>−</sup> <sup>C</sup>k||<sup>F</sup> , in Figure [6,](#page-17-0) of all linaer layers in RoBERTa-base by sweeping the number of iterations. We observe that the weight approximation error monotonically decreases with the number of iterations, but as shown in Figure [1,](#page-6-2) the model output error may increase. This observation indicates that the commonly used objective of minimizing the weight approximation error and the corresponding algorithm are not ideal for the quantization error reconstruction problem.

