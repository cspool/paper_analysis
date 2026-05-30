# Limitation

Although showcasing superior performance, the preprocessing scheme still has limitations to be reckoned with, which requires more runtime to get a start point before quantization. For example, our runtime reaches 2h on LLaMA-7B, and fortunately, this falls within an acceptable range (OmniQuant reports 1.1h but exhibits worse performance and higher bit-width per weight). Considering that extremely low-bit quantization is the most challenging quantization scenario especially for PTQ, we believe it is worthwhile to sacrifice some computational resources within an acceptable range to pursue higher performance.

In addition, due to the limitation that commercial NVIDIA GPUs do not support such low-bit inference, and designing specific hardware requires larger research teams and financial support, we cannot provide real-world inference evaluation results yet. Our goal is to explore the performance limits

of PTQ by fake-quantization before commercial hardware support is available. We believe this will eventually be realized as evidenced by the quick development of GPUs.

## Ethics Statement

This paper introduces solutions to the challenges associated with Large Language Models (LLMs) quantization, with the overarching goal of facilitating the widespread adoption and application of LLMs. In the current landscape, ethical concerns tied to LLMs, including the presence of hidden biases encoded in the models, are garnering heightened attention. Following our investigation, we assert that our proposed method does not further amplify the biases and contravene any ethical standards.

## Acknowledgment

Miao Zhang was partially sponsored by the National Natural Science Foundation of China under Grant 62306084 and U23B2051, and Shenzhen College Stability Support Plan under Grant GXWD20231128102243003 and Grant ZDSYS20230626091203008.

