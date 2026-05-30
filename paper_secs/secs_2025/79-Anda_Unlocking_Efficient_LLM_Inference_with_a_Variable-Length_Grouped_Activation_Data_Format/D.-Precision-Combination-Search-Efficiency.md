# D. Precision Combination Search Efficiency

Our algorithm aims to efficiently optimize FP activations in weight-only quantized LLMs during the post-training phase.

![](_page_5_Figure_6.jpeg)

Fig. 9. Search process of the proposed adaptive precision combination search algorithm on the OPT-125M model with constraint under 1% accuracy loss, which efficiently finds the global optimum within 10 iterations.

Most weight-only quantization processes [24], [51], [66] rely on a small calibration dataset, which we can reuse in the activation precision search. Ensuring a rapid search process is critical to avoid extending post-training deployment time. Therefore, the algorithm is designed to find a near-optimal solution quickly, within an acceptable accuracy tolerance, to enable efficient hardware deployment.

The efficiency of our algorithm is enhanced by two key mechanisms: First, we introduce a constraint that updates the best combination only when a new precision combination offers a lower computational cost, employing a relaxation strategy similar to gradient descent to accelerate convergence. While this may miss the global optimum, it ensures a high-performance combination within limited iterations. Second, we set an iteration limit to complete the search within a reasonable timeframe, avoiding deployment delays. It is here important to note that the relatively limited search space of only 4 precision variables allows for fast convergence with just a few iterations. The execution time of each iteration is roughly the time of a forward pass over the calibration dataset to validate the precision combination.

To demonstrate our algorithm's search efficiency, we compare it with the conventional brute-force approaches [12]–[14] on the OPT-125M model. As shown in Fig. 9, the search space for OPT-125M contains over 10,000 possible combinations, and our algorithm identifies the precision combination [7, 7, 6, 5] in just 10 iterations, maintaining accuracy within 1% loss. In practice, we limit the search to 32 iterations, ensuring that time overhead remains minimal while achieving a near-optimal precision combination. By avoiding time-consuming backward propagation or complex solving processes, our algorithm operates approximately twice as fast as Omniquant [66] and ten times faster than GPTQ [24], the current SoTA methods for post-training weight-only LLM quantization.

