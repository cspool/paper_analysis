# III. ANDA DATA FORMAT

In this section, we present unique features of the Anda data format and demonstrate its benefits towards FP-INT operations in weight-only quantized LLM inference. Furthermore, we introduce a mantissa bit-width search method to efficiently

TABLE I Anda format definition in contrast with prior BFP formats

| Multi-Length | 2b/4b<br>2b/4b/8b<br>4b/8b/16b        | Chunk-serial<br>Bit-parallel                                     | BFP Chunk-based<br>BFP Element-based<br>BFP Element-based |  |
|--------------|---------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------|--|
| Multi-Length |                                       |                                                                  | BFP Chunk-based                                           |  |
|              | 2b/4b                                 | Chunk-serial                                                     |                                                           |  |
|              |                                       |                                                                  | DIT Element based                                         |  |
|              | 16b                                   |                                                                  | BFP Element-based                                         |  |
| Uni-Length   | 15b                                   |                                                                  | BFP Element-based                                         |  |
|              | 14b                                   | Dit-paranei                                                      | FP16 Element-based                                        |  |
|              | 8b                                    | Dit porollal                                                     | BFP Element-based                                         |  |
|              | 5b                                    |                                                                  | BFP Element-based                                         |  |
|              | 4b                                    |                                                                  | BFP Element-based                                         |  |
| BFP Type     | Mantissa Length<br>during Computation | Computation                                                      | Storage                                                   |  |
|              |                                       | during Computation  4b 5b Uni-Length Bright during Length 4b 14b | Ab   Sb   Bit-parallel                                    |  |

identify the optimized Anda precision combinations that satisfy a user-defined accuracy drop.

#### A. Anda Format Features

Based on the findings of our previous study, we propose the Anda format: an innovative variable-length mantissa BFP scheme designed for efficient LLM inference. Anda's structure comprises a sign bit, a shared exponent, and a variable-length mantissa, building upon traditional BFP conversion processes as previously shown in Fig. 4. Its key feature is the ability to dynamically select mantissa lengths for different tensors based on their precision sensitivity, maintaining consistency within each tensor while optimizing the accuracy-efficiency trade-off.

Table I compares Anda with prior BFP formats, categorizing them based on supported mantissa lengths. Uni-length formats, such as VS-Quant [12] and FIGNA [32], use fixed mantissa lengths, while multi-length formats like FAST [85] and Da-Capo [41] offer limited flexibility with 2~3 predefined lengths. Anda surpasses both by providing a continuous range of mantissa lengths, allowing fine-grained precision control across different LLM modules. Enabled by specialized hardware units, as detailed in Sec. IV, smaller mantissa widths result in a lower inference latency, computational cost and memory storage cost. This allows Anda format to carefully balance model precision and computational efficiency, providing a more aggressive compression in less sensitive model parts while preserving critical precision elsewhere.

#### B. Efficient FP-INT GeMM Using Anda Format

We then compare the workflows of GeMM workloads of several SotA approaches to illustrate the advantages of replacing FP16 activations with the Anda data format. Taking the W4A16 quantization scheme as an example, we examine the FP-INT GeMM computation process (a) on existing GPU platforms [52]; (b) on GPU platforms with dedicated FP-INT processing units; (c) using FIGNA's dynamic conversion scheme [32]; and (d) with our proposed Anda approach. Fig. 8 depicts the four schemes, with colors indicating the data types used throughout the computational process.

Fig. 8(a) shows the workflow of W4A16 LLMs on common GPU platforms. The absence of dedicated FP-INT computation units in GPU necessitates converting INT4 weights to FP16 before processing, with tensor cores operating in FP16 mode. This scheme not only brings additional format conversion overheads, but requires costly FP computations.

![](_page_4_Figure_0.jpeg)

Fig. 8. Comparison of (a) the current computation scheme on GPU, (b) and that enhanced with dedicated FP-INT processing unit, (c) FIGNA scheme, and (d) our Anda scheme for FP-INT GeMM. Our Anda scheme significantly reduces memory space, access cost, and computation cost and enables energy-efficient precision-scalable operations.

GPU platforms equipped with dedicated FP-INT processing units, as illustrated in Fig. 8(b), can eliminate the need for converting INT4 weights to FP16, thereby reducing data conversion overheads and computation costs. However, as pointed out by FIGNA [32], the high alignment and normalization overhead associated with FP-INT processing units still results in high computational expenses.

To efficiently deploy W4A16 LLMs, FIGNA proposes a computation scheme using a BFP variant with corresponding hardware support to overcome the issues with dedicated FP-INT units. As depicted in Fig. 8(c), activations are stored in FP16 format in memory, converted to the FIGNA format before computation, after which a 14-bit mantissa is multiplied with INT4 weights for GeMM computation. The final results are then converted again to FP16 and written back to memory. This scheme reduces the computation overhead by converting costly FP GeMM to INT operations. However, since FP16 activations need to be repeatedly accessed during computation, frequent data conversion from FP16 to FIGNA introduces additional overhead, affecting overall efficiency.

As presented in Fig. 8(d), our proposed Anda format computation scheme offers some unique advantages in contrast with the previous approaches. Firstly, the activations are no longer stored in memory in FP16 format, but directly in the Anda data format, reducing storage overhead and data access overhead while avoiding frequent data conversion. Secondly, the shared exponent enables INT dot-product operations within a group, followed by FP32 accumulation across groups, reducing the computational overhead of FP-INT GeMMs. Thirdly, the variable-length mantissa considerably decreases dot-product operations and memory accesses use the minimal necessary word length. Finally, converting only the final FP32 results back to Anda format before writing to memory minimizes the storage requirement and the additional overhead from switching data format.

#### C. Adaptive Precision Combination Search

To leverage the Anda format for fast deployment and hardware performance gains, we propose an adaptive preci-

sion search algorithm for offline compile-time optimization of activation precisions in weight-only quantized LLMs. Our algorithm is built around two key strategies. (a) We narrow the search space to the precision of only four key tensor types ,i.e.,  $A_{qkv}$ ,  $A_o$ ,  $A_u$ , and  $A_d$ , based on their sensitivity to model accuracy as demonstrated in Fig. 7. This precision combination is represented as a 4-tuple  $[M_{qkv}, M_o, M_u, M_d]$ . (b) We employ a training-free, one-shot calibration process reusing the small amount of calibration data from the post-training weightonly quantization process, being several thousands of tokens with hundred batches [24], [51], [66]. Though prior layer-wise methods [18], [28], [76] may achieve finer precision adjustments, their prolonged search times significantly extend the deployment process. In contrast, our module-wise approach rapidly assigns mantissa lengths while maintaining consistency across layers and can easily be integrated into standard posttraining deployment workflows.

As outlined in Algorithm 1, we take the LLM model L, a calibration dataset D, an accuracy loss tolerance  $\delta$ , and a maximum number of iterations N as inputs. The accuracy tolerance  $\delta$  specifies the acceptable level of performance degradation, while the maximum number of iterations N serves as a termination criterion, ensuring the algorithm concludes within a reasonable time frame. With these inputs, our algorithm finds the optimal 4-tuple precision combination within the given iterations that best balances model accuracy and inference efficiency across the model's key activation components. The search process consists of three key steps.

Step 1: Initialize search starting points. A priority queue with precision combinations of equal precision across all modules is initialized first. These precision combinations range from aggressive (e.g., [4,4,4,4]) to conservative (e.g., [13,13,13,13]). This strategy enables the rapid discovery of efficient combinations while ensuring the existence of feasible solutions, as validated by our prior experiments in Fig. 6.

**Step 2: Check the promising combination.** In each iteration, the combination with the lowest bit operations (BOPs) is extracted from the priority queue and added to the visited

#### **Algorithm 1:** Adaptive Precision Combination Search

```
Input: LLM model L, calibration dataset D,
            accuracy loss tolerance \delta, max iterations N
    Output: Optimized precision combination best_comb
              denoted as a 4-tuple [M_{qkv}, M_o, M_u, M_d]
    // S1: Initialize search starting points
1: Q \leftarrow PriorityQueue([4, 4, 4, 4], ..., [13, 13, 13, 13]);
2: best\_comb \leftarrow null, best\_bops \leftarrow \infty;
3: iterations \leftarrow 0, visited \leftarrow {};
4: fp\ acc \leftarrow EvaluateAccuracy(L, D);
5: while iterations < N do
          / S2: Check the promising combination
        bops\_eval \leftarrow Q.map(EvalBOPs);
        curr\_bops \leftarrow min(bops\_eval);
        curr\_comb \leftarrow Q.get(bops\_eval.index(curr\_bop));
        visited \leftarrow visited \cup \{curr\_comb\};
        anda\_acc \leftarrow \textbf{EvaluateAccuracy}(L, D, curr\_comb);
10:
            S3: Update and relax the best combination
11:
        if curr\_bops < best\_bops and
         anda\_acc \ge (1 - \delta) \cdot fp\_acc then
             best\_comb \leftarrow curr\_comb;
12:
             best\_bops \leftarrow curr\_bops;
13:
             neighbors \leftarrow GenerateCandidates(curr\_comb);
14:
             foreach n \in neighbors do
15:
                 if n \notin visited then
16:
                     Q.push(n);
17:
                 end
18:
             end
19:
20:
        end
21:
        if Q.empty() then
            break;
22:
        end
23.
        iterations \leftarrow iterations + 1;
24:
25: end
   return best_comb
```

set. The BOP metric [1], [43], [49], [71] quickly estimates computational cost by calculating the total number of bit operations for the necessary multiplications under a given combination. This allows us to efficiently prioritize promising combinations without a full model evaluation. The accuracy of the promising combination is then examined on the calibration dataset.

Step 3: Update and relax the best combination. If the evaluated combination yields lower BOPs than the current best while maintaining accuracy within the specified tolerance, it becomes the new best combination. To generate nearby precision candidates, the algorithm then relaxes this best combination by decreasing the mantissa length of each tensor type by one, while keeping the other tensor types unchanged. For example, if the current best combination is [6,7,5,5], the generated candidates will be [5,7,5,5], [6,6,5,5], [6,7,4,5], and [6,7,5,4]. The generated candidates that have not been visited before are added to the priority queue. If the accuracy constraint is not met, no update is made. Step 2 and 3 are repeated until the maximum number of iterations is reached or the search space is exhausted.

