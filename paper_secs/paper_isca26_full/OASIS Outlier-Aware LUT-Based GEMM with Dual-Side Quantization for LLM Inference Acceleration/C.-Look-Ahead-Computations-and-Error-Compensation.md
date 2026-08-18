# C. Look-Ahead Computations and Error Compensation

As discussed in §II-C, it is important to effectively hide the latency of outlier detection. To address this challenge, we propose a look-ahead computation and error compensation design

that concurrently performs WAQ LUT-GEMM computations and outlier detection. Fig. 7 shows an example of the proposed look-ahead computation and error compensation design, which are performed in two parallel branches: the main branch (left) and the outlier branch (right). In this M-K-N GEMM example,  $M=1,\,K=6,\,N=4,$  and  $n_W=n_A=1.$  There are two activation outliers in channels 2 and 4, indicated by deep green boxes.

- 1) Look-Ahead Computations: In the main branch, the entire FP16 activation vector is clustered to the nearest centroids in the activation codebook  $\mathcal{C}_A$ , producing quantized activations. While the quantization error of the inliers is generally small and does not lead to significant accuracy degradation, the quantization error of the outliers is non-negligible. For instance, the activation outliers in channels 2 and 4, originally 5.07 and -3.01, are quantized to 0.78 and 0.13, yielding quantization errors (residuals) of 4.29 and -3.14, respectively. Our proposed look-ahead computation design allows the main branch to temporarily ignore the quantization errors of outliers and directly perform WAQ LUT-GEMM computations using the quantized activations, while the outlier branch simultaneously computes the error compensation for the outliers.
- 2) Error Compensation: In the outlier branch, the outlier detection units dynamically identify the outliers in the activation vector and compute their quantization errors by subtracting the quantized activation outliers from the original FP16 activation outliers. We use the channel index of the activation outliers to fetch the corresponding input channels of quantized weights from the  $idx_W$  matrix, which are then dequantized to FP16 format based on the weight codebook  $\mathcal{C}_W$ . Then, the residuals are multiplied by the dequantized weights to generate error compensation terms, which are accumulated into the look-ahead computation results using WAQ LUT-GEMM from the main branch. As shown in Fig. 4, this mechanism removes the time-consuming dynamic outlier detection from the critical path, enabling reduced GEMM runtime and ensuring identical mathematical results.

Note that in each cycle, the outlier detection unit sequentially outputs each pair of outlier value and their channel index. Thus, in each cycle, only one input channel of the weight index matrix is fetched and dequantized for error compensation. Compared to the conventional design for dynamic outlier identification (Fig. 4(a)) which performs sparse FP16 GEMMs for all residuals simultaneously, this design eliminates the need for sparse representation of the outliers and significantly reduces the number of multiply-accumulate (MAC) units required in the outlier branch.

#### IV. HARDWARE DESIGN

To enable the efficient execution of the computation optimizations introduced in §III, in this section, we present the supporting accelerator design of OASIS. First, we present the overall architecture of the OASIS accelerator. Then, we describe the micro-architecture designs of the Index Counter and the Clustering Unit, which are the key hardware components to support the proposed WAQ LUT-based GEMM. Moreover,

![](_page_5_Figure_6.jpeg)

Fig. 8. Overall architecture of the OASIS accelerator.

we also propose a lightweight outlier detection engine *Orizuru*, which dynamically identifies the top-k largest and smallest elements in each activation token with minimal overhead.

#### A. Overall Architecture

Fig. 8 shows the overall architecture of the OASIS accelerator, and Table II presents its hardware configurations. The OASIS accelerator is composed of 16 Processing Element (PE) Lines, a LUT, an Activation Index Buffer, an Output Buffer, Orizuru units, Clustering Units, Functional Units, an Error Calculation Unit, and a Memory Controller. Besides the Cartesian Product results, the LUT also stores the centroid codebooks of activations and weights for the purposes of activation clustering and weight dequantization, respectively. The main compute unit within the PE Lines is the Concat Unit, which accepts two 4-bit indices, concatenates them, and stores the result in an 8-bit register. The Concat Unit's minimalist design provides high area efficiency compared to FP16 MAC units, benefiting both compute-intensive and memoryintensive scenarios. For compute-intensive workloads such as LLM prefill phase, this efficiency enables a higher number of Concat Units on the chip, delivering the computational parallelism needed for high throughput. For memory-intensive workloads such as LLM decode phase, the lightweight Concat Units consume minimal chip area, freeing up chip area for additional I/O pins to increase bandwidth and mitigate memory bottlenecks.

Within each PE Line, there are 4096 Concat Units, a Weight Index Buffer, 32 Index Counters, a Dequantization Unit, a MAC Tree for reduction, and 8 MAC Units for error compensation. The computation flow of the main and outlier branches is illustrated with the red and black circled numbers, respectively.

For the main branch, in step ①, the Clustering Units reads the entire activation vector in the Output buffer, and clusters them to the nearest centroids based on the Activation Codebook. The clustered activation indices are then stored in the Activation Index Buffer. In step ②, the clustered activation indices are fetched from the Activation Index Buffer and broadcast to all PE Lines. In step ③, the Concat Units in each PE Line concatenate the activation indices with the weight

![](_page_6_Figure_0.jpeg)

Fig. 9. Design of (a) the Index Counter and (b) the Clustering Unit.

indices fetched from the Weight Index Buffer to form the concatenated indices. Next, in step **<sup>4</sup>** , the Index Counters calculate the counts of the unique concatenated indices. Finally, in step **<sup>5</sup>** , the MAC Tree performs the weighted-sum operations based on the counts and the corresponding Cartesian Product values retrieved from the LUT, generating the look-ahead computation results.

For the outlier branch, in step **<sup>1</sup>** , the *Orizuru* units read the FP16 activations in the Output buffer and dynamically identify the outliers, sequentially outputting each outlier value along with its channel index. The weight input channel corresponding to the outlier channel index is fetched from the Weight Index Buffer in step **<sup>2</sup>** , and dequantized to FP16 weights in the Dequantization Unit in step **<sup>3</sup>** . In step **<sup>4</sup>** , the Error Calculation Unit calculates the residual between the outlier activation and its nearest centroid from the Activation Codebook. Step **<sup>4</sup>** is performed in parallel with **<sup>2</sup>** and **<sup>3</sup>** . Then, in step **<sup>5</sup>** , the MAC Units perform multiply-accumulates between the outlier residuals and the dequantized weights, generating the error compensation terms. Finally, the MAC units combine the error compensation terms with the look-ahead computation results from the main branch to produce the final output activations, which are stored back in the Output Buffer.

To minimize end-to-end latency, the Memory Controller orchestrates pipelined execution of operations across both the main and outlier branches. The cycle latencies for each computation step are presented in § V-D3.

