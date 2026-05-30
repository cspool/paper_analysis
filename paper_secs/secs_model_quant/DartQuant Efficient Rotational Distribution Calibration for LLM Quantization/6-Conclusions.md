# 6 Conclusions

In this paper, we introduce DartQuant, an innovative method for LLM quantization that efficiently handles activation outliers. By constraining the activation distribution, DartQuant simplifies rotation matrix calibration and avoids overfitting risks in end-to-end fine-tuning. The QR-Orth optimization method eliminates the need for complex orthogonal optimization, further speeding up the process. DartQuant achieves state-of-the-art results in 4-bit quantization while significantly reducing costs. Notably, it successfully quantizes a 70B model on a single RTX 3090, advancing LLM deployment in resource-constrained environments.

