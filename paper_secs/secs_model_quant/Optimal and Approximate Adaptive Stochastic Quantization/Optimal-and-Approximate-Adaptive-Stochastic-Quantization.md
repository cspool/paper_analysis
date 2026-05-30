# Optimal and Approximate Adaptive Stochastic Quantization

Ran Ben Basat UCL Yaniv Ben-Itzhak VMware Research Michael Mitzenmacher Harvard University **Shay Vargaftik** VMware Research

## **Abstract**

Quantization is a fundamental optimization for many machine learning (ML) use cases, including compressing gradients, model weights and activations, and datasets. The most accurate form of quantization is adaptive, where the error is minimized with respect to a given input rather than optimizing for the worst case. However, optimal adaptive quantization methods are considered infeasible in terms of both their runtime and memory requirements.

We revisit the Adaptive Stochastic Quantization (ASQ) problem and present algorithms that find optimal solutions with asymptotically improved time and space complexities. Our experiments indicate that our algorithms may open the door to using ASQ more extensively in a variety of ML applications. We also present an even faster approximation algorithm for quantizing large inputs on the fly.

