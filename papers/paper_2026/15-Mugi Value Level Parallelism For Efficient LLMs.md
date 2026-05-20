**==> picture [136 x 44] intentionally omitted <==**

## **Mugi: Value Level Parallelism For Efficient LLMs** 

## Daniel Price 

## Prabhu Vellaisamy 

daniel.price@ucf.edu pvellais@andrew.cmu.edu University of Central Florida Carnegie Mellon University Department of ECE Department of ECE Orlando, FL, USA Pittsburgh, PA, USA 

## **Abstract** 

Value level parallelism (VLP) has been proposed to improve the efficiency of large-batch, low-precision general matrix multiply (GEMM) between symmetric activations and weights. In transformer based large language models (LLMs), there exist more sophisticated operations beyond activation-weight GEMM. In this paper, we explore _how VLP benefits LLMs_ . First, we generalize VLP for nonlinear approximations, outperforming existing nonlinear approximations in end-toend LLM accuracy, performance, and efficiency. Our VLP approximation follows a value-centric approach, where important values are assigned with greater accuracy. Second, we optimize VLP for small-batch GEMMs with asymmetric inputs efficiently, which leverages timely LLM optimizations, including weight-only quantization, key-value (KV) cache quantization, and group query attention. Finally, we design a new VLP architecture, Mugi, to encapsulate the innovations above and support full LLM workloads, while providing better performance, efficiency and sustainability. Our experimental results show that Mugi can offer significant improvements on throughput and energy efficiency, up to 45× and 668× for nonlinear softmax operations, and 2 _._ 07× and 3 _._ 11× for LLMs, and also decrease operational carbon for LLM operation by 1 _._ 45× and embodied carbon by 1 _._ 48×. 

_**CCS Concepts:**_ • **Hardware** → **Application-specific VLSI designs** ; **Emerging architectures** ; _Arithmetic and datapath circuits_ ; • **Computer systems organization** → **Parallel architectures** . 

_**Keywords:**_ value-level parallelism, value reuse, data reuse, computation reuse, unary computing, temporal coding, quantization, general matrix multiplication, nonlinear approximation, large language model, KV cache, group query attention 

## **ACM Reference Format:** 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu. 2026. Mugi: Value Level Parallelism For Efficient LLMs. In _Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS_ 

This work is licensed under a Creative Commons Attribution 4.0 International License. 

_ASPLOS ’26, Pittsburgh, PA, USA_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2359-9/2026/03 https://doi.org/10.1145/3779212.3790189 

## John P. Shen 

## Di Wu 

jpshen@cmu.edu 

jpshen@cmu.edu di.wu@ucf.edu Carnegie Mellon University University of Central Florida Department of ECE Department of ECE Pittsburgh, PA, USA Orlando, FL, USA 

**==> picture [240 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
Activation Weight Operations beyond GEMM<br>4<br>DNN<br>FP8 FP8 ReLU GEMM Nonlinear<br>3 2 1<br>LLM<br>BF16 INT4 SiLU/GELU softmax/exp KV cache<br>**----- End of picture text -----**<br>


**Figure 1.** Challenges for LLM inference using VLP. 

_’26), March 22–26, 2026, Pittsburgh, PA, USA._ ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/3779212.3790189 

## **1 Introduction** 

Modern artificial intelligence (AI) has been betting on deep neural networks (DNNs) for over a decade [35]. A great deal of software and hardware research has been dedicated to improving the efficiency of GEMM, as it accounts for over 90% of the total runtime [31]. A key insight from this body of work is that low numerical precision gives good efficiency and accuracy. Previous research has proposed and applied narrower data formats symmetrically to the activations and weights of GEMM, e.g. BF16 [32], DLFloat16 [2], CBFloat [51], FP8 [41]. For offline workloads that process _large-batch_ , _lowprecision_ data, value level parallelism (VLP) can potentially improve performance and efficiency by avoiding redundant computations, as shown by a Carat design [46], with details given in Section 2.1. 

**Challenges.** Recent AI advancements have given rise to generative AI workloads, e.g., transformer-based large language models (LLMs) [64]. LLMs exhibit complicated operations beyond activation-weight GEMM, for which VLP is designed. Naturally, a research question is raised: _can VLP address diverse LLM operations?_ Figure 1 highlights the challenges. 

1 First, prior VLP architectures do not support LLM nonlinear operations, such as such as SiLU [17], Swish [56], GELU [25], and softmax. These operations are far more complicated than the ReLU predecessor [20] and account for significant runtime if not optimized [9, 34, 55, 61, 72], despite various software [45, 68] and hardware [27, 42, 66, 67, 72] solutions being proposed. 

1216 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

2 Second, prior VLP architectures misalign the trending asymmetric quantization in LLM inference, offering suboptimal efficiency. Memory-intensive LLMs outgrow the memory capacity in mobile devices easily [16, 43, 49]. BF16-INT4 quantization has been exploited on both weight [19] and KV cache [26] to combat the large memory footprint while preserving accuracy. But prior Carat only supports FP8 [46]. 

3 Third, prior VLP architectures are not optimized to for small-batch inputs, leading to suboptimal efficiency. LLMs usually use a small batch size, such as 8, ensure real-time inference, since large batch sizes linearly worsen the inference latency [1, 81] and violate the system-level objectives (around 200 ms) [10, 59], 

4 Fourth, existing AI architectures dedicate separate matrix and vector units for nonlinear operations and GEMM, increasing the carbon emission and lowering sustainability. The nonlinear hardware increases on-chip area and embodied carbon during manufacture, which could outweigh the operational carbon during LLM execution, especially for more advanced technologies [22]. 

**Proposal.** To overcome the challenges above, we craft Mugi, a new VLP architecture that support nonlinear approximation for the first time and asymmetric, small-batch GEMM, as well as reusing the array for both nonlinear operations and GEMM for efficient LLMs. First, Mugi orchestrates the first-to-date VLP support for nonlinear approximation. Mugi approximates critical nonlinear operations in LLMs, such as softmax, SiLU, and GELU. Mugi adopts _input approximation_ and generates a precise output for an approximate input, in contrast to common output approximation with precise input [27, 42, 66, 67, 72]. VLP approximation is _value centric_ and assign greater accuracy to more important inputs. 

Moreover, Mugi is optimized for asymmetrically quantized, small-batch GEMMs, that are not compatible in prior VLP designs [46]. LLMs leverages weight-only quantization (WOQ) [8, 13, 19, 28, 38, 39] for activation-weight GEMM and KV cache quantization (KVQ) [26, 33, 60, 82, 83] for activation-activation GEMM, introducing BF16-INT4 GEMM. Mugi supports such asymmetric quantization by _customizing the data format_ and _optimized mapping_ . This optimization ensures high utilization for both WOQ with small-batch input and KVQ with grouped query attention (GQA) [3]. We additionally _minimize the buffer cost_ in Mugi over Carat via broadcasting and output buffer leaning. 

Last but not least, Mugi synergizes the nonlinear approximation and GEMM optimizations and maximally reuse the chip budget for LLMs. This allows Mugi to execute full LLM workloads efficiently and decrease area overhead, both of which directly correlate to a decrease in operational and embodied carbon. 

The contributions of this paper are summarized as follows: 

- We formulate value level parallelism for nonlinear approximation, which adopts input approximation in a value-centric manner. 

- We optimize value level parallelism asymmetric, smallbatch GEMM, using timely LLM optimizations, such as quantization and group query attention. 

- We synergize the nonlinear approximation and GEMM optimization above in one Mugi architecture to run full LLM workloads. 

- We conduct experiments on multiple LLMs using Mugi and demonstrate good improvements in performance, efficiency and sustainability. 

This paper is organized as follows. Section 2 reviews the background. Section 3 articulates VLP approximation. Section 4 describes our Mugi architecture, with evaluations in Section 5 and Section 6. Section 7 and Section 8 discuss and conclude this paper. 

## **2 Background** 

## **2.1 Value Level Parallelism** 

Value level parallelism (VLP) was first proposed for GEMM operations on large-batch, low-precision data [46], with an example for vector-scalar multiplication and vector-vector outer product given in Figure 2. (a) shows the temporal coding of a variable _𝑖_ , done by a temporal converter (TC) in green, which is essentially an equivalence logic. When input _𝑖_ , of value 3, equals the number in counting-up sequence (i.e., when the counter _𝑐_ reaches 3 in the rectangle), the TC asserts a temporal spike in red at the 3rd cycle; otherwise, no spikes is generated, indicated by the bold segments in black. (b) and (c) depicts transforming a multiplication between _𝑖_ = 3 and _𝑤_ = 1 into the accumulation of _𝑤_ over time. At cycle _𝑖_ , the accumulation outputs the correct product _𝑖𝑤_ , equivalent to 1 + 1 + 1 = 3. (d) exemplifies _temporal subscription_ . Saving the correct product (Val) into the register in yellow is enabled by the temporal spike (Sub), essentially selecting its corresponding _𝑖𝑤_ product of 3 × 1 = 3 in red. (e) extends to scalar-vector multiplication between a scalar _𝑤_ and a vector � _𝑖_ . The accumulation results of _𝑤_ are shared by all vector elements. Each vector element subscribes to its own product in parallel, based on it own input (red and blue). We call such parallel, value-dependent sharing across multiple inputs as _value reuse_ . Value reuse and temporal subscription together formulate VLP. (f) shows that a vector-vector outer product can be obtained by organizing multiple columns of scalarvector multiplication into a 2D array. Carat maps batched input activations to rows and weights to columns, with the number of columns matching the temporal spike latency to avoid resource overprovision and maximize the resource utilization. Since the temporal spike latency increases exponentially, 2 _[𝑛]_ cycles for an _𝑛_ -bit input, it is more beneficial to keep VLP at smaller bitwidths [74, 80]. Therefore, Carat opts 

1217 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**==> picture [240 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) i=3 = i=3 = i=3 =<br>c=0 c=1 c=3<br>(b)w=1i=3 x 3 (c)w ACC 543210 (d) 5 4 3 2 1 0 Val = iw<br>w=1 ACC<br>(e) 5 4 3 2 1 0 Sub<br>Val i=3 TC 3<br>w=1 ACC iw (f) Weights<br>i[0]=3 TC 3 ACC ACC ACC<br>i[1]=1 TC 1 TC<br>TC<br>i[k]=3 TC 3 TC<br>…<br>… …<br>Batched<br>activations<br>**----- End of picture text -----**<br>


**Figure 2.** Illustration of VLP, detailed in Section 2.1. 

for mapping the batch dimension to rows to achieve scalable performance with large batch sizes. 

## **2.2 Nonlinear Implementations** 

**2.2.1 Software Implementation.** We take nonlinear operations in LLMs such as softmax, SiLU [17], and GELU [25] as examples, formulated in Equations 1, 2, and 3 [47], where erf means error function. To ensure numerical stability by avoiding overflow in exp, softmax inputs are usually subtracted by the maximum of all inputs. Without KV cache, softmax can take more than 40% of the total runtime in transformer models [9, 34, 55, 61]. The GELU function is commonly approximated as shown in Equation 4 or Equation 5 [47]. The functions easily take tens even hundreds of cycles to finished [45, 68]. 

**==> picture [236 x 50] intentionally omitted <==**

**==> picture [211 x 14] intentionally omitted <==**

**2.2.2 Piecewise Linear Hardware Approximation.** To ensure high efficiency, hardware approximations are proposed. Piecewise linear (PWL) approximation [27, 67] separates the function curve into multiple linear segments based on the input range and computes the result based on which segment an input falls into. PWL approximations need to buffer the segment coefficients and identify the corresponding segment of an input via comparison. For an input vector, a dedicated set of buffers, comparators, and arithmetic units is needed for each element, increasing hardware overheads. 

**2.2.3 Taylor Series Hardware Approximation.** Another popular hardware approximation is a Taylor series [42, 66, 71, 72, 77]. The coefficient of each Taylor term is precomputed. With Horner’s rule, the computation can be transformed in to concatenated multiply-accumulate (MAC) operations [72]. This transformation allows for vectorized implementation, where coefficients can be shared by all inputs efficiently. However, Taylor approximation exhibits poor accuracy when 

inputs are far from the Taylor expansion point. Allowing more points introduces addition hardware overheads. 

**Summary.** Different implementations offer distinct accuracy and efficiency tradeoffs, and this work introduces a novel VLP approximation for nonlinear operations. 

## **2.3 Large Language Model Inference** 

Modern LLMs are mainly built on attention-based transformers [64]. During prefilling, multiple tokens are processed in parallel, resulting in GEMM operations. During decoding, one token is processed, resulting in GEMV operations, unless input tokens are batched. However, even with batched input, normal attention still performs GEMV for KV cache [50]. 

**2.3.1 Grouped Query Attention.** GEMV in normal attention severely lowers the hardware utilization [83]. To mitigate the problem, grouped query attention (GQA) is proposed, where multiple Q tokens share the same KV cache [3], creating small-batch GEMM. In this paper, Mugi benefits from GQA to improve the hardware utilization. 

**2.3.2 Weight-Only Quantization.** Low-bit quantization is now the de facto technique to reduce the memory footprint [36]. Most prior works adopt symmetric quantization, e.g., INT8 or FP8 for both inputs and weights [12, 16, 36, 44], which are still too memory inefficient for LLMs. Developers resort to sub-byte quantization for weights, while keeping the inputs in floating point format, e.g, BF16-INT4 weight only quantization (WOQ) [8, 13, 19, 28, 38, 39]. 

**2.3.3 KV Cache Quantization.** KV cache introduces additional memory footprint on top of weights, leading to potential out-of-memory errors [83]. BF16-INT4 KV cache quantization (KVQ) has been leveraged to compress KV cache with minimum accuracy drop [26, 33, 60, 82, 83]. Moreover, WOQ and KVQ can be combined together, reporting just a 0.02 increase in perplexity [26]. 

**Summary.** These LLM optimizations are timely and allow optimizing asymmetric, small-batch GEMM efficiently. 

## **2.4 Sustainable Computing** 

Carbon emissions have become a growing concern in the AI world [29] and AI inference reportedly contributes up to 90% of datacenter costs [6]. To quantify carbon emissions, carbon modeling focuses on _operational_ carbon for workload deployment and _embodied_ carbon for infrastructure manufacture over the full lifetime [18, 48, 70]. The equivalent emissions (CO2eq) are formulated in Equation 6, where E, CI, CPA are short for energy and carbon intensity, carbon emitted per unit area. Embodied carbon is taking over operational carbon as the majority of contributed emissions [18, 70]. 

**==> picture [221 x 9] intentionally omitted <==**

1218 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

**Summary.** Our Mugi shares the hardware for both nonlinear operation and GEMM, and contributes to reduction of both operational and embodied carbon. 

can be shared across the array in parallel, individually subscribing to their final result. 

## **3.2 Input Approximation** 

## **3 VLP Nonlinear Approximation** 

## **3.1 Formulation** 

We formulate VLP approximation as in Figure 3. (a) depicts a conventional lookup table (LUT) for exp, indexing with an address and receiving its corresponding value. (b) indicates that such a conventional LUT can only sequentially process different inputs, limiting the scalability. To alleviate this restriction, (c) splits the lookup processes into two steps. First, a row of exp values with consecutive input values are retrieved from the LUT, then the correct value can be selected from that row. Such a split inspires VLP approximation. (d)(e) illustrates this split within VLP. (d) uses an input sign and mantissa (i: S-M) to index the LUT row, and (e) then uses the exponent (i: E) to select the final result from the row. 

(f) details VLP approximation within a single row. 1 – 4 denotes four phases, i.e., input field split, value reuse, mantissa temporal subscription, and exponent temporal subscription. 1 the input field split phase splits the input S-M-E (0-3-2) into S-M (0-3) and E (2), as in (d)-(e). 2 the value reuse phase organizes the LUT the same as that in (d), where each LUT row contains all values for the same S-M. At each cycle, an ascending address is sent to the LUT, and generates an output LUT row, marked by bold lines. Overtime, LUT rows are reused by different S-M values. 3 the mantissa temporal subscription phase further splits the S-M to reuse LUT rows. S-M (0-3) generates a temporal signal via the temporal converter in green, subscribing to the row at the 3rd cycle with a red fill. The LUT row will be stored into the yellow blocks when a temporal spike arrives. 4 the exponent temporal subscription leverages the temporal spike of the exponent (E=2) to subscribe the final exp results from the LUT row, selecting the value at the 2nd cycle with a blue outline. _Starting from the moment_ when the correct LUT row is subscribed in 3 , the exponent also starts generating its own temporal spikes. Therefore, the full VLP approximation requires the total duration of both mantissa and exponent temporal spike timing to finish. 

(g) zooms into the single row for mantissa and exponent temporal subscription. The LUT row subscription is indicated by S-M. ‘-x’ here denotes all exponents. The corresponding rows will be sent to the proper inputs, indicated by the bold lines. The exponent subscription is indicated by the blue outlined blocks. Following (f), this example selects the row corresponding to an S-M of (0-3), subscribing at the 3rd cycle, then subscribing where E=2, or at the 2nd cycle of 4 in (f). This full process takes a total of 6 cycles, which is the sum of two subscription. 

(h-i) expands approximation to a full array, enabling approximation of vector _𝑖_[�] . By leveraging VLP, selected rows 

VLP approximation favors lower-precision inputs to reduce the duration of temporal spikes. However, popular data formats have a wide mantissa field, e.g., BF16 mantissa has 7 bits. Therefore, in the input field split phase, we round the input mantissa to fewer bits for softmax, SiLU and GELU. Profiling shows that rounding introduces uniform errors to the input mantissa values, as the mantissa are uniformly distributed in softmax, SiLU and GELU, and this pattern is consistent across models and modalities. We exclude these results for simplicity. 

## **3.3 Value-Centric Approximation** 

The approximation efficiency also suffers from wide exponents, since both the temporal signal length and the LUT row size grows exponentially with exponent bitwidth. Our formulation leverages the insights that input exponents are often clustered at a small range. We focus on these important values, following a value-centric approach. We profile the input distribution of softmax, SiLU and GELU in Figure 4. For softmax, exponent values are concentrated around [−3 _,_ 4], despite input values being widely spread. Similar observations also exist in SiLU and GELU. We define these model-specific, important exponents as the LUT window, and we only store the results for these exponents in the LUT. However, a single mapping, with a set of inputs for value reuse, might not cover the full range of important exponents. Therefore, we opt for a sliding window for each mapping and choose an optimal range, as shown in Figure 5. 

## **3.4 Accuracy Impact** 

We explore the accuracy of different window sizes and boundaries as in Figure 6. We compare VLP approximation against prior approximations, like Taylor series [42, 66], piecewise linear (PWL) approximation [67], and partial approximation (PA) [27]. For most models with full VLP approximation (combined softmax/activation), Mugi shows better accuracy, except for Llama 2, whose softmax distribution varies significantly across layers, as shown in Figure 4. To address this, Figure 7 demonstrates per-layer tuning of Llama 2, selecting the optimal LUT range for each layer. This mitigates accuracy loss, yielding perplexity values approaching those in line with other approximation techniques. 

Figure 8 further shows the accuracy of the nonlinear approximation. While VLP approximation does not exhibit the best error, it has the best accuracy where inputs are important, in term of magnitude and quantity. For softmax in layer 0, high exp accuracy for majority of the inputs (Figure 4) propagates less errors to deeper layers. In deeper layers, more inputs center around −10. Their output magnitudes are smaller, e.g, adding 22k _𝑒_[−][10] equals _𝑒_[0] . Therefore, even 

1219 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**==> picture [504 x 253] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) LUT (b) LUT (g)<br>exp(0) i[0]=0 [Addr] exp(0) Val  exp(0) Cycle 7 6 5 4 3 2 1 0<br>… … LUT row 0-4-x 0-3 -x 0-2-x 0-1-x<br>Val  i[1]=2 exp(2)<br>i=2 [Addr] exp(2 ) exp(2) exp(2) i: E=2 0-3- 2 0-3-1 0-3-0<br>… …<br>exp(4) i[k]=4 exp(4) exp(4)<br>… … Time<br>(c)i={0,1,2} [Addr] exp(0)exp(1)LUT exp(2) Val  LUT row (f)1 i: S-M=0-3Addr LUTTC 3 S-M=0-3Val2 LUT row i[04]: E=2 E=2 exp S-M-E0-3-2<br>exp(3)exp(4)exp(5) exp(0)exp(1)exp(2)<br>…<br>(i) Cycle 7 6 5 4 3 2 1 0<br>LUT row 0-4-x 0-3 -x 0-2-x 0-1 -x<br>(d)<br>LUT: exp(S-M-E) i[0]: E=2 0-3- 2 0-3-1 0-3-0<br>i: S-M=0-0 Addr 0-0-00-1-0 0-0-10-1-1 0-0-20-1-2 Val  0-0-0LUT row0-0-1 0-0-2 i[1]: E=1i[k]: E=0… 0-3-2 0-3-1 0-3-0-1-2 0 0-1- 1 0-1-0<br>… Time<br>(e) (h) Addr LUT Val2 LUT row i[04]: E=2 E=2 exp S-M-E<br>S-M-E i[0]: S-M=0-3 TC 0-3-2<br>i: E=0 Addr 0-0-0LUT row0-0-1 0-0-2 Val  exp(0-0-0) i[1]: S-M=0-1 TC 3 i[1]: E=1 0-1-1<br>1 S-M=0-3 …<br>i[k]: E=0<br>i[k]: S-M=0-3 TC 0-3-0<br>… …<br>…<br>…<br>**----- End of picture text -----**<br>


**Figure 3.** VLP approximation for nonlinear operations, exp here, with a floating-point input _𝑖_ , represented as S-M-E, denoting the sign, mantissa and exponent. More details are in Section 3.1. 

**==> picture [505 x 172] intentionally omitted <==**

**----- Start of picture text -----**<br>
Llama 2 Whisper SwinV2 ViViT<br>7B 13B Tiny Large Tiny Large Base<br>2 SM 4 SM 4 SM 1 SM<br>1 2 2<br>0 0 0<br>-16 -8 0 -16 -8 0 -16 -8 0 -12 -6 0 -16 -8 0 -16 -8 0 -12 -6 0<br>40 S G 10 G G<br>2 1<br>20 5<br>0 0 0 0<br>-2 0 2 -8 0 8 -10 -5 0 -4 0 -8 0 8 -8 0 8 -10 -5 0<br>50 SM 60 SM 50 SM 40 SM<br>25 30 25 20<br>0 0 0 0<br>-8 0 8 -8 0 8 -8 0 8 -8 0 8 -8 0 8 -8 0 8 -8 0 8<br>30 S 60 G 60 G 40 G<br>15 30 30 20<br>0 0 0 0<br>-8 0 8 -8 0 8 -8 0 8 -8 0 8 -8 0 8 -8 0 8 -8 0 8<br>Value<br>Exp<br>**----- End of picture text -----**<br>


**Figure 4.** Distribution of input values and exponents of nonlinear operations in transformer models. Profiled layers, stages, and sequence lengths are detailed in Table 1. Cooler colors represent early layers, while warmer colors represent later layers. Within each color, lighter lines represent shorter sequence lengths, while darker lines represent longer sequence lengths. Softmax, SiLU, and GELU are abbreviated as SM, S, and G, denoting the nonlinear function within each window. 

though VLP approximation is less accurate, it has negligible impacts compared to inputs closer to 0. Another contributor to overall accuracy is that uniform input errors from input approximation can cancel out each other’s output errors during summation. For SiLU/GELU, inputs consistently cluster around 0, where VLP approximation is more accurate. 

## **4 Mugi Architecture** 

We introduce Mugi, a novel VLP architecture to support nonlinear approximation and asymmetric, small-batch GEMM, as well as reusing the array for both nonlinear operations and GEMM for efficient LLMs. Mugi supports nonlinear softmax, SiLU and GELU operations. Mugi supports BF16-INT GEMM, leveraging timely GQA, WOQ and KVQ optimizations. The support for both nonlinear operations and GEMM 

1220 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

**==> picture [178 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
LUT window (S-M)<br>E-6 E-5 E-4 E-3 E-2 E-1 E0 E1 E2 E3 E4 E5<br>Slide window (S-M)<br>E-3 E-2 E-1 E0 E1 E2 E3 E4<br>i[0]: S-M TC<br>i[1]: S-M TC<br>…<br>i[k]: S-M TC<br>…<br>…<br>**----- End of picture text -----**<br>


**Figure 5.** An example sliding window for input mapping. This example chooses the exponent range of [−3 _,_ 4] for the current set of inputs, from the full LUT window with the exponent range of [−6 _,_ 5]. The sliding window size of 8 is chosen to match the VLP array width in prior VLP works [46]. This sliding window can slide left and right for each mapping, aiming to minimize the accuracy loss. 

above are mixed in one singular architecture with maximized resource reuse. Figure 9 outlines our Mugi architecture. It double buffers all memory hierarchies to hide access latency. We follow prior VLP designs and set the number of columns to 8 (matching 3-bit mantissa) for optimal performance and efficiency tradeoffs [46]. 

1 is for the input field split phase. M-proc splits the sign (S) and mantissa (M) fields for BF16 input and approximates the mantissa to 3-bit via rounding (R). For a given mapping, E-proc processes the exponent (E) values to determine the maximum or minimum exponent, which determines the LUT sliding window in the SW block. The sliding window size is fixed to 8 to match array width. It also clamps the exponent, underflowing to 0, and overflowing depending on the nonlinear operation. In softmax, overflow values are set to the maximum value of the LUT, while SiLU/GELU passes values through directly. The exponent is sent to post processing (PP) block for the final result. 

2 is for the value reuse phase. The iSRAM acts as the LUT, and the pre-computed, output LUT window is sent to the SW block to generate the sliding window. The sliding window is sent to iFIFO to stagger the input by one cycle to adjacent columns. This staggering ensures fully pipelined execution in our VLP approximation. 

3 is for the mantissa temporal subscription phase. The temporal converter (TC) converts the approximate mantissa (M) to a temporal signal using the counter (CNT) and leaves the sign (S) to PP. The temporal signal is then pipelined in a row via the T register in the processing element (PE). Temporal subscription is done using the AND gate in each PE. Within a PE column (Figure 9 (d)), both the counter value and sliding window are broadcast. Within a PE row (Figure 9 (e)), the subscribed results will be sent out via OR gates, since only one column will be activated by the pipelined temporal spike. Two sets of OR gates, together with a small FIFO, 

double buffer the results from two spikes. Sign conversion (SC) XORs all signs to generate the final result. 

4 is for the exponent temporal subscription phase. The PP block takes the exponent from the E-proc and generates a MUX selection signal. If no special values exist, this selection signal is the temporal spike from the exponent, subscribing the correct element in the sliding window. If there are special values, the multiplexer outputs the proper special values among Zero, infinity (INF) and Not-a-Number (NaN). 

## **4.1 Nonlinear Approximation** 

To better understand how Mugi works, we give a walkthrough example in Figure 10. Here, rows apply broadcasting, while columns adopt pipelining. The LUT (iSRAM) stores pre-computed nonlinear results, and each LUT row contains a vector of results for one mantissa. The LUT size will double if the nonlinear operation has both positive and negative inputs. In 0 – 7 , TC or PP is red if a temporal signal (yellow) arrives. In 8 – 9 , a new mapping is marked with blue. 

In the first input field split phase, the 8-bit BF16 mantissa is rounded to INT4 with a 3-bit mantissa magnitude to generate an 8-cycle temporal signal. In the second value reuse phase, LUT row vectors are read out per cycle in a mantissaascending order, and reused in the next subscription phase. Note that vectors from different LUT columns are sent to the array in a staggered manner, as the temporal signal is pipelined across columns. In the third mantissa temporal subscription phase, temporal signals are generated from approximated mantissa. For each mantissa, the TC turns red upon the coincidence of the input value and equivalent clock cycle. The fourth exponent temporal subscription phase subscribes to the correct result from the LUT vector, indicated by the red exponent (e). The sign (s) is omitted here as it is always negative. The cycle index to get the final result is the sum of mantissa and exponent values. After 8 cycles of red input, at cycle 8, new blue inputs enter the array. 

The above VLP approximation works for element-wise nonlinear operations, e.g., exp and SiLU/GELU. Additional summation and division are needed for softmax. We first compute the exp for all inputs (maximum subtracted). To perform the summation, when we compute exp, the output accumulator (oAcc) simultaneously accumulates the exp results. Once all exp operations finish, we store the sum to the oSRAM from the oFifo. Next, we divide all exp results by the sum using the vector multiplication array (Vec) in Figure 9. This array multiplies the exp by the reciprocal of the sum in one cycle. To ensure high utilization, we map both attention head and batch across rows for softmax. 

## **4.2 GEMM Optimization** 

Mugi optimizes GEMM over prior VLP designs in two ways. **Format Customization.** Asymmetric BF16-INT4 GEMM imposes challenges to Carat. As Carat maps FP8 input across 

1221 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**==> picture [505 x 271] intentionally omitted <==**

**----- Start of picture text -----**<br>
Llama 2 7B Llama 2 13B Whisper Tiny Whisper Large SwinV2 Tiny SwinV2 Large ViViT<br>8 6.71 6.28 6.89 6.93 6.71 5.99 6.53 6.54 20.3 12.0 11.1 4.90 4.42 4.00 3.39 0.90 0.86 0.82 0.94 0.95 0.71 0.69 0.68 0.69 0.76 3.38 2.00 1.76 1.77 1.77<br>9 6.72 6.28 6.89 6.91 6.71 5.99 6.53 6.54 20.3 12.0 11.1 4.90 4.42 4.00 3.39 0.90 0.86 0.82 0.94 0.95 0.71 0.69 0.68 0.69 0.76 3.38 2.00 1.76 1.77 1.77<br>10 6.72 6.28 6.89 6.90 6.71 5.99 6.53 6.54 20.3 12.0 11.1 4.90 4.42 4.00 3.39 0.90 0.86 0.82 0.94 0.95 0.71 0.69 0.68 0.69 0.76 3.38 2.00 1.76 1.77 1.77<br>11 6.72 6.28 6.89 6.90 6.71 5.99 6.53 6.54 20.3 12.0 11.1 4.90 4.42 4.00 3.39 0.90 0.86 0.82 0.94 0.95 0.71 0.69 0.68 0.69 0.76 3.38 2.00 1.76 1.77 1.77<br>12 6.72 6.28 6.89 6.90 6.71 5.99 6.53 6.54 20.3 12.0 11.1 4.90 4.42 4.00 3.39 0.90 0.86 0.82 0.94 0.95 0.71 0.69 0.68 0.69 0.76 3.38 2.00 1.76 1.77 1.77<br>0 1 2 3 4 0 1 2 3 4 -2 -1 0 1 2 -2 -1 0 1 2 -2 -1 0 1 2 -2 -1 0 1 2 0 1 2 3 4<br>8 6.37 5.76 5.76 5.76 5.81 6.02 5.21 5.20 5.20 5.23 22.4 22.4 22.4 22.5 3.52 5.11 5.10 6.78 4.44 0.82 0.92 0.91 0.72 0.72 0.70 1.23 7.34 6.54 1.75 1.73 1.76 1.76<br>9 6.37 5.76 5.76 5.76 5.76 6.02 5.21 5.20 5.20 5.20 22.4 22.4 22.4 22.4 3.52 5.12 5.11 6.61 4.94 0.87 0.91 0.91 0.71 0.72 0.73 1.57 7.45 6.27 1.87 1.75 1.77 1.77<br>10 6.37 5.76 5.76 5.76 5.76 6.02 5.21 5.20 5.20 5.20 22.4 22.4 22.4 22.4 3.53 5.13 5.11 6.71 3.59 0.89 0.91 0.91 0.71 0.72 0.71 1.35 7.21 5.26 1.84 1.75 1.76 1.77<br>11 6.37 5.76 5.76 5.76 5.76 6.02 5.21 5.20 5.20 5.20 22.4 22.4 22.4 22.4 3.54 5.09 5.14 6.22 1.91 0.88 0.90 0.90 0.70 0.71 0.68 1.03 6.21 3.86 1.83 1.76 1.76 1.78<br>12 6.37 5.76 5.76 5.76 5.76 6.02 5.21 5.20 5.20 5.20 22.4 22.4 22.4 22.3 3.53 5.08 5.10 5.49 1.88 0.91 0.94 0.93 0.74 0.75 0.79 1.57 5.06 3.88 2.05 1.97 2.04 1.93<br>0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 -2 -1 0 1 2 -8 -7 -6 -5 -4 -3 -2 -1 0 1 -7 -6 -5 -4 -3<br>20 5.76 5.76 5.76 5.76 5.75 5.19 5.19 5.19 5.19 5.19 5.15 5.20 5.11 5.15 5.11 0.87 0.87 0.92 0.91 0.90 0.69 0.70 0.72 0.71 0.71 1.77 1.77 1.78 1.78 1.77<br>22 5.76 5.76 5.76 5.75 5.75 5.19 5.19 5.19 5.19 5.19 23.8 5.19 5.11 5.14 5.19 5.12 0.93 0.90 0.87 0.90 0.92 0.72 0.70 0.70 0.71 0.73 1.77 1.77 1.77 1.78 1.77<br>23 5.76 5.76 5.76 5.75 5.75 5.19 5.19 5.19 5.19 5.19 23.0 24.0 5.09 5.12 5.20 5.09 5.08 0.87 0.90 0.90 0.91 0.89 0.70 0.71 0.71 0.72 0.71 1.78 1.77 1.78 1.77 1.77<br>24 5.76 5.76 5.75 5.75 5.75 5.19 5.19 5.19 5.19 5.19 22.6 5.12 5.17 5.00 5.11 5.12 0.90 0.91 0.90 0.92 0.94 0.71 0.72 0.71 0.72 0.73 1.77 1.78 1.77 1.77 1.77<br>25 5.76 5.75 5.75 5.76 5.75 5.19 5.19 5.19 5.19 5.19 24.9 5.14 5.08 5.09 5.14 5.05 0.93 0.90 0.91 0.92 0.91 0.73 0.71 0.72 0.72 0.71 1.78 1.77 1.77 1.78 1.77<br>-20 -19 -18 -17 -16 -20 -19 -18 -17 -16 -21 -20 -19 -18 -17 -20 -19 -18 -17 -16 -24 -23 -22 -21 -20 -24 -23 -22 -21 -20 -24 -23 -22 -21 -20<br>20 5.79 5.77 5.79 5.84 5.90 5.32 5.21 5.21 5.25 5.29 22.5 22.9 23.0 23.7 24.8 5.36 5.43 5.26 5.18 5.13 0.91 0.91 0.91 0.92 0.92 0.72 0.72 0.72 0.71 0.72 1.77 1.78 1.78 1.79 1.79<br>22 5.78 5.78 5.78 5.80 5.82 5.32 5.21 5.21 5.22 5.23 22.4 22.9 23.5 23.5 24.2 5.28 5.32 5.00 4.93 5.26 0.91 0.90 0.91 0.91 0.91 0.72 0.72 0.72 0.72 0.73 1.77 1.77 1.77 1.78 1.80<br>23 5.78 5.77 5.78 5.82 5.94 5.32 5.21 5.21 5.23 5.29 22.4 22.9 23.0 23.3 24.2 5.36 5.33 5.25 5.19 4.98 0.91 0.90 0.91 0.91 0.93 0.72 0.72 0.72 0.72 0.71 1.77 1.77 1.78 1.78 1.78<br>24 5.78 5.76 5.78 5.79 5.82 5.32 5.20 5.21 5.21 5.23 22.4 22.7 23.2 23.7 23.9 5.25 5.34 5.37 5.25 5.47 0.90 0.91 0.90 0.91 0.91 0.73 0.72 0.72 0.73 0.72 1.77 1.77 1.77 1.78 1.80<br>25 5.78 5.76 5.77 5.82 5.84 5.32 5.20 5.21 5.23 5.24 22.4 22.9 23.0 22.9 23.3 5.24 5.35 5.34 5.31 5.19 0.91 0.91 0.90 0.91 0.92 0.72 0.72 0.73 0.73 0.72 1.77 1.77 1.78 1.78 1.78<br>3 4 5 6 7 3 4 5 6 7 3 4 5 6 7 7 8 9 10 11 3 4 5 6 7 7 8 9 10 11 3 4 5 6 7<br>6 5.94 5.84 5.81 5.81 5.88 5.35 5.29 5.27 5.28 5.40 22.9 22.4 22.9 5.11 4.45 4.69 4.89 5.04 0.86 0.85 0.84 0.84 0.86 0.70 0.70 0.69 0.69 0.69 1.83 1.78 1.76 1.76 1.76<br>7 5.84 5.80 5.78 5.80 5.87 5.29 5.26 5.23 5.27 5.37 22.5 22.2 22.4 4.44 4.67 4.91 5.01 5.05 0.85 0.84 0.84 0.86 0.88 0.70 0.69 0.69 0.69 0.70 1.78 1.76 1.76 1.76 1.76<br>8 5.80 5.79 5.78 5.80 5.84 5.26 5.24 5.24 5.26 5.32 22.2 22.4 21.7 4.67 4.89 4.99 5.05 5.08 0.84 0.84 0.86 0.88 0.89 0.69 0.69 0.69 0.70 0.70 1.76 1.76 1.76 1.76 1.76<br>9 5.79 5.78 5.78 5.78 5.82 5.25 5.24 5.24 5.24 5.30 22.2 22.5 21.8 4.86 4.97 5.06 5.08 5.11 0.84 0.85 0.87 0.89 0.90 0.69 0.69 0.70 0.70 0.71 1.76 1.76 1.76 1.76 1.76<br>10 5.78 5.78 5.78 5.78 5.81 5.24 5.24 5.24 5.23 5.27 21.9 22.5 21.8 4.96 5.04 5.06 5.08 5.10 0.85 0.87 0.89 0.90 0.91 0.69 0.70 0.70 0.71 0.71 1.76 1.76 1.76 1.76 1.77<br>-7 -6 -5 -4 -3 -7 -6 -5 -4 -3 -4 -3 -2 -1 0 -9 -8 -7 -6 -5 -9 -8 -7 -6 -5 -9 -8 -7 -6 -5 -9 -8 -7 -6 -5<br>Base VLP PWL T Base VLP PWL T Base VLP PWL T Base VLP PWL T Base VLP PWL T Base VLP PWL T Base VLP PWL T<br>5.75 6.21 5.68 5.78 5.19 6.00 5.21 5.23 22.23 11.1 22.2 21.68 5.06 3.52 5.05 4.44 0.91 0.83 0.91 0.84 0.71 0.67 0.71 0.69 1.77 1.75 1.77 1.76<br>VLP SM<br>LUT Size<br>Min/Max Exp<br>VLP S/G<br>PWL SM<br>Segments<br>Segment Range<br>PWL S/G<br>Degrees<br>Taylor SM Degree Center<br>Full PPL<br>**----- End of picture text -----**<br>


**Figure 6.** Perplexity and loss heatmaps of transformer models, showing the best for each nonlinear operation separately highlighted in orange (lower is better). Llama 2 and Whisper show perplexity, while Swin and Vivit show Loss. Combined softmax/activation metrics are short right of the title, with VLP being on top and PWL on bottom. Boxes on the left denote the approximation method, with softmax, SiLU and GELU abbreviated as SM, S, and G. The labels next to the boxes denote y axis value and _the labels on the right side denote x axis_ . LUT size refers to the number of exponents stored in the LUT, and Min/Max exp denotes the maximum or minimum value used to create the LUT. For PWL, segment range (sr) denotes the approximation range, with SM going from [sr, 0] and S/G going from [-sr, sr]. For taylor series, degrees denotes the number of polynomial expansions used, while degree center is the center of the expansion. Empty boxes represent masked large values. Each column’s table lists full end-to-end perplexity values (SM and S/G). The Taylor-series values, denoted T, include only SM. 

**==> picture [242 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
6.2<br>6.0<br>5.8 Llama 2 7B    Final PPL: 5.98<br>Llama 2 13B    Final PPL: 5.43<br>5.6<br>5.4<br>0 5 10 15 20 25 30 35 40<br>PPL<br>**----- End of picture text -----**<br>


**Figure 7.** Per-layer perplexity tuning of Llama 2 (7B, 13B), with final achieved perplexity noted in the legend. Tuning is done progressively across layers. 

rows, changing input to BF16 will prolong the temporal signal from 8 (3-bit mantissa magnitude) to 128 cycles (7-bit), prohibitively lowering the throughput. If INT4 weights were mapped to rows, the FP8 datapath is wasted. To overcome these, Mugi transposes the mapping, i.e., INT4 weight/KV cache to rows with a slim INT4 datapath, and BF16 input/Q token to columns. This mapping offers high utilization, since 

LLM tokens are large enough to fill in all rows, and 8 Q tokens in a group for GQA can fill in all columns. The customization timely synergizes with the trends of LLMs, small batch sizes, large token sizes, WOQ, KVQ and GQA, none of which are compatible in Carat. WOQ and KVQ require dequantization after GEMM, which is done by the vector array. 

**Buffer Minimization.** Buffers (FIFOs) occupy significant area in Carat, due to pipelining the inputs across rows and double buffering in the output OR tree. The relevant cost scales quadratically with the array size. Mugi solves this problem via broadcasting and output buffer leaning (optimizing two FIFOs into one without functional changes), successfully lowering the total buffer area by 4 _._ 5×. 

With the optimizations above, GEMM can be exectued efficiently, following the flow in Figure 2. To ensure scalability, we can further use a 2D mesh Network-on-Chip (NoC) to connect multiple nodes. We consider output stationary dataflow and inter-node accumulation, and GEMMs are evenly tiled across nodes to enhance efficiency and utilization. 

1222 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

**==> picture [241 x 185] intentionally omitted <==**

**----- Start of picture text -----**<br>
Exp PWL Exp Taylor Exp Mugi<br>0% 6% 0% 1% 100% -0.5 0 1%0%<br>-50% -50% 0% -2%<br>-100% 0% -0.5 0 -100% -3% -0.5 0 0%<br>16 8 0 8 4 0 16 8 0<br>SiLU PWL SiLU PA SiLU Mugi<br>100%<br>0% -0.5 0.5<br>22% 0% 10% 50% 5%0%<br>-50% 0% 0% -6%<br>-100% -22%-0.5 0.5-100% -7% -0.5 0.5 0%<br>5 0 5 5 0 5 5 0 5<br>GELU PWL GELU Mugi<br>100% -0.5 0.5<br>5%<br>0% 34% 50% 0%<br>-6%<br>0%<br>-100% -0.5 0. 5-34% 0%<br>5 0 5 5 0 5<br>**----- End of picture text -----**<br>


**Figure 8.** Relative errors against software implementation. The most accurate configurations from Figure 6 are compared. x and y axes are input values and relative error. 100% error indicates flushing output to 0. 

**Table 2.** Comparison of Mugi and baselines: SA refers to systolic array, SD to SIMD array, Tensor to Tensor Core, with off-chip bandwidth at 256 GB/s. Input (i), weight (w), and output (o) refer to respective components, with ranges (e.g., a-b) covering all powers of 2 between a and b. Input word config applies to the query word, and weight word to key and value words with kvcache quantization. -S denotes scaled-up configurations. All designs use 4x4 and 8x8 NoC layouts except Tensor and -S configurations, which use a 2x1 and 2x2 and no NoC, respectively. 

|**Confguration**|**Mugi**<br>**Carat**<br>**SA**<br>**SD**<br>**SA-S**<br>**SD-S**<br>**Tensor**|**Mugi**<br>**Carat**<br>**SA**<br>**SD**<br>**SA-S**<br>**SD-S**<br>**Tensor**|**Mugi**<br>**Carat**<br>**SA**<br>**SD**<br>**SA-S**<br>**SD-S**<br>**Tensor**|**Mugi**<br>**Carat**<br>**SA**<br>**SD**<br>**SA-S**<br>**SD-S**<br>**Tensor**|
|---|---|---|---|---|
|i/w/o SRAM<br>Array height (H)<br>Array width (W)<br>Array Depth (D)<br>Input word<br>Weight word<br>NoC shape|64KB||1MB||
||32 - 256<br>8|4 - 16<br>32 - 64<br>H||16<br>8|
|||N/A||16|
|||16<br>4|||
||4x4, 8x|8|N/A|2x1, 2x2|



## **5.2 Hardware** 

**Table 1.** Studied LLMs in this paper. 

|**Model**<br>**7**|**Llama2 [63]**<br>**W**|**Llama2 [63]**<br>**W**|**hisper [54]**<br>**Swin**|**V2**<br>**ViViT**|
|---|---|---|---|---|
||**B**<br>**13B**|**70B**<br>**ti**|**ny**<br>**large**<br>**tiny**|**large**<br>**base**|
|Batch size<br># layers<br>3<br># stages<br># attn heads<br>3<br># KV heads<br>3<br>Attn h dim<br>40<br>FFN h dim<br>11<br>Seq len|1 - 32||||
||2<br>40<br>2<br>40<br>2<br>40<br>96<br>5120<br>008<br>13824|80<br>4<br>64<br>6<br>8<br>6<br>8192<br>38<br>28672<br>15|32<br>12<br>4<br><br>20<br>3 - 24<br><br>20<br>3 - 24<br>4<br>1280<br>96-768<br>36<br>5120<br>384-3072|24<br>12<br>4<br>6 - 48<br>12<br>6 - 48<br>12<br>192-1536<br>768<br>768-6144<br>3072|
||4096||1500<br>64-4|096<br>3136|
|Prof. layers<br>1/1<br>Prof. seq len|6/32<br>1/20/40|1/32/64<br>1/3|/6<br>1/10/20<br>1/12|1/24<br>1/6/12|
||1024<br>2048<br>4096||112/224<br>16/3<br>375/750<br>64/10<br>1500<br>2048/4|2<br>784<br>24<br>1568<br>096<br>3136|



* _ℎ_ = hidden, Prof. = Profiled 

## **5 Experimental Setup** 

## **5.1 Large Language Model** 

The evaluated LLMs are summarized in Table 1, with LLaMA270B supporting GQA using a group size of 8. We implement all models using the HuggingFace Transformers Package [69], profiling and computing loss or perplexity for each model over 100 inferences. During profiling, we extract the runtime input nonlinear tensors across all tokens and record the value and exponent distributions, which is documented in Figure 4. To obtain the perplexity and loss values shown in Figure 6, we sweep various configurations for each nonlinear implementation, and show windows containing the best performing configuration. For both figures, we show the smallest and largest model of each model family, with the only exception of Llama 2 13B replacing Llama 2 70B due to memory and runtime limitations. 

**5.2.1 Mugi.** As shown in Table 2, we set the oSRAM width to enable wFIFO loading of nonlinear operations in 8 cycles, ensuring sufficient bandwidth. The wSRAM width is similarly configured to allow loading in 8 cycles for GEMM operations. Mugi’s vector array is configured to scale array outputs after exiting the oFIFO, hiding latency. Mugi is optimized for output stationary outer-product computation. 

**5.2.2 Baseline.** We build baselines with components for both nonlinear operations and GEMM. 

Nonlinear approximation hardware uses alternative vector arrays with added components to implement the Taylor series and PWL approximation methods [42, 67]. The Taylor series is implemented with Horner’s method up to 9 degrees, and requires additional registers to store coefficients. PWL implementation adopts 22 segments, requiring additional registers and comparators to store and select proper segments. Both methods are configured to achieve their best perplexity as shown in Figure 6. We also consider a vector array of MAC units to precisely compute nonlinear operations, which require 44 cycles [45, 68]. 

For GEMM, we compare Mugi with Carat [46], systolic arrays, SIMD arrays, FIGNA [30], and tensor cores from Nvidia Hopper GPUs [43], as well as a Mugi-L design. Given Carat only supports FP8 GEMM with inputs mapped across rows, we modify its accumulators at the top to BF16 and map inputs across columns, while using its FP8 data path for INT4 weights. The systolic and SIMD arrays are similar, with the systolic array needing additional control hardware and a column of output accumulators, compared to SIMD’s adder trees. Like Mugi, Carat implements output stationary, while the systolic and SIMD arrays use weight stationary configurations. The FIGNA configurations extend both systolic 

1223 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**==> picture [502 x 113] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 (b) (d) CNT iFIFO (f) M-proc (g)<br>iSRAM iFIFO iFIFO iFIFO SW Max/Min<br>T M R E-proc<br>(a) 3 CNT iAcc iAcc iAcc E-proc T C PE S E Clam p<br>PE<br>T C PE<br>wFIFO T C PE PE PE PP oAcc oFIFO T C PE (h) PP<br>M-sel<br>wFIFO T C PE PE PE PP oAcc oFIFO (c)<br>TC (e)<br>wFIFO T C PE PE PE PP oAcc oFIFO = T C PE PE PE Zero<br>1 4 M INF<br>M-proc oSRAM Vec S SC NaN<br>wSRAM<br>Off-chip memory<br>**----- End of picture text -----**<br>


**Figure 9.** Overview of Mugi architecture. 1 – 4 (yellow, red, green, and blue) correspond to 1 – 4 phases of VLP approximation in Figure 3. Purple is for mapping softmax, and gray marks the additional hardware for GEMM. 

**==> picture [504 x 223] intentionally omitted <==**

**----- Start of picture text -----**<br>
a S=0/1 M7E0 M7E1 … M7E7 0 S=1 0, 0 … 1 S=1 1, 0 0, 1 … 2 S=1 2, 0 1, 1 …<br>M E M E M E<br>LUT M1E0 M1E1 … M1E7 0 T C 1 PP 0 T C 1 PP 0 T C 1 PP<br>M0E0 M0E1 … M0E7 1 T C 2 PP 1 T C 2 PP 1 T C 2 PP<br>M S, E 0 T C 2 PP 0 T C 2 PP 0 T C 2 PP<br>m 0 T C PP 4 T C PP 4 T C PP 4 T C PP<br>s0e0 2 2 2<br>m 1 T C PP<br>s1e1<br>m 2 T C PP<br>s2e2 3 S=1 3, 0 2, 1 … 4 S=1 4, 0 3, 1 … 5 S=1 5, 0 4, 1 …<br>m 3 T C PP<br>s3e3 M E M E M E<br>0 T C PP 0 T C PP 0 T C PP<br>1 1 1<br>1 T C PP 1 T C PP 1 T C PP<br>2 2 2<br>0 T C PP 0 T C PP 0 T C PP<br>2 2 2<br>4 T C PP 4 T C PP 4 T C PP<br>2 2 2<br>6 S=1 6, 0 5, 1 … 0, 6 7 S=1 7, 0 6, 1 … 1, 6 0, 7 8 S=1 0, 0 7, 1 … 2, 6 1, 7 9 S=1 1, 0 0, 1 … 3, 6 2, 7<br>M E M E M E M E<br>0 T C PP 0 T C PP 1 T C PP 1 T C PP<br>1 1 1 1<br>1 T C PP 1 T C PP 0 T C PP 0 T C PP<br>2 2 2 2<br>0 T C PP 0 T C PP 0 T C PP 0 T C PP<br>2 2 2 2<br>4 T C PP 4 T C PP 2 T C PP 2 T C PP<br>2 2 2 2<br>…<br>… … … …<br>…<br>… … …<br>… … … …<br>**----- End of picture text -----**<br>


**Figure 10.** Mapping element-wise nonlinear operations to Mugi. a abstracts the VLP array for nonlinear operations. M, S and E are for mantissa, sign and exponent. 0 – 9 are examples for exp, and the numbers denotes the clock cycles. 

and SIMD arrays with the FIGNA PE, which is customized for FP-INT multiplication. Additionally, scale-up versions of both systolic and SIMD arrays with MAC and FIGNA configurations are evaluated. The tensor core has GEMM shaped as 8x16x16, and is a fully pipelined design to perform 8x16x16 MAC operations per cycle. We only compare tensor core in NoC settings. The Mugi-L uses a dedicated LUT to approximate nonlinear operations, rather than temporal coding based approximation. We ensure 8 inputs share one LUT to match the throughput of Mugi. Similar to Mugi, all design’s wSRAM and oSRAM widths are selected to load the array without introducing additional latency. 

**5.2.3 Network on Chip and Off-Chip Memory.** The NoC incorporates three channels for input, weight, and output. An output-stationary approach is employed across all implementations. Each design operates with a NoC frequency of 400 MHz and a HBM bandwidth of 256 GB/s from off-chip 

memory. Both the NoC and off-chip memory are configured such that they always supply the minimum bandwidth required to not bottleneck computation. 

## **5.3 Carbon Modeling** 

To model both Mugi and baselines operational and embodied carbon, we follow a consistent approach by previous works [18, 48, 70]. Operational carbon is computed as the product of E and CI, while embodied carbon is the product of Area and CPA, both outlined in Section 2.4. For CI, we use world carbon intensity outlined in ACT [23]. We compute CPA with _𝐸_ / _𝑚𝑚_[2] detailed in Dark Silicon [7], and convert it to CPA with the previously stated CI. 

## **5.4 Simulation** 

We developed an in-house architecture simulator based on a publicly available artifact for Carat [46]. We extend the 

1224 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

**==> picture [240 x 180] intentionally omitted <==**

**----- Start of picture text -----**<br>
  Mugi  Carat  VA-FP    VA-AP<br>SM (128) SM (128) SM (16) SM Taylor (16)<br>SiLU (128) SiLU (128) SiLU (16) SM PWL (16)<br>SM (256) SM (256) SM T (16) SiLU PWL (16)<br>SiLU (256) SiLU (256) SiLU T (16)<br>Norm Throughput<br>48x<br>32x<br>16x<br>Norm Energy Efficiency<br>750x<br>500x<br>250x<br>Norm Power Efficiency<br>18x<br>12x<br>6x<br>128 256 512 1024 2048 4096<br>**----- End of picture text -----**<br>


**Figure 11.** Iso-area comparison between Mugi and baselines for nonlinear operations, across sequence lengths (128–4096) with a batch size of 8, geometric meaned across all Llama 2 models (7B, 13B, 70B). SM, and VA-FP, and VA-AP abbreviate softmax, precise vector array, and approximate vector arrays. Titles in the legend indicate the array type of each column. All results are normalized to VA (16) with array heights indicated in “()”. 

artifact and build an event-based simulator that can hierarchically solve the mapping of nonlinear operations and GEMM on our customized Mugi architecture and report area, leakage power, dynamic energy, cycle count, and runtime. The basic hardware modules are implemented in RTL and synthesized under 400MHz with 45nm technology to retrieve metric values. The memory access power are obtained from CACTI7 [5]. 

We also place and route the full RTL of a single node 8x8 Mugi at 400MHz, and obtain area 0.056 _𝑚𝑚_[2] and frequency of 408.5MHz with critical path on VLP broadcast. We further increase the synthesis frequency of Mugi, and reach 975MHz without timing violations. We stick to 400MHz to isolate of the impact of the implementation during the evaluation. 

## **6 Evaluation** 

## **6.1 Nonlinear Approximation** 

**6.1.1 Accuracy.** As shown in Figure 6, Mugi softmax, SiLU, and GELU implementations achieve highly accurate results, outperforming other methods on most models. Even in cases where Mugi is not the most accurate, Llama 2, it remains comparable to prior implementations. Conversely, when value distributions are concentrated, Mugi yields substantial improvements in performance, as evident by the perplexity gains observed in Whisper Tiny and highlighted in Figure 8. In contrast, prior approximation methods [27, 42, 67] do not consider the value distribution of workloads prior to approximation, introducing additional error. 

**==> picture [241 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
Mugi (128) Carat (128) SA (16) SD (16)<br>Mugi (256) Carat (256) SA-F (16) SD-F (16)<br>Projection/FFN Attention<br>2x 10x<br>1x 5x<br>3x 20x<br>2x<br>10x<br>1x<br>2x 4x<br>1x 2x<br>7B 13B 70B 7B 13B 70B GQA<br>Throughput<br>Energy Eff<br>Power Eff<br>**----- End of picture text -----**<br>


**Figure 12.** Iso-area comparison for projection (proj), attention (attn), and feed-forward network (ffn) GEMM operations in Llama 2 (7B, 13B, 70B, and 70B with GQA). All results are normalized to that of 16×16 systolic array (SA). -F denotes FIGNA. Batch size is set to 8, and sequence length is set to 4096, with array heights indicated in parenthesis (). 

**6.1.2 Throughput and Efficiency.** Figure 11 compares the throughput and efficiency. Since all designs map the head dimension across rows, sequence length does not impact the normalized performance gains. Mugi achieves a shared normalized throughput of 45×, and energy and power efficiency improvements of 481 _._ 07× and 10 _._ 69× for softmax, and 667 _._ 85× and 14 _._ 84× for SiLU compared to precise vector arrays. Mugi outperforms PWL approximation in softmax by 5× (throughput), 8 _._ 53× (energy efficiency), and 1 _._ 71× (power efficiency), and in SiLU by 5×, 10 _._ 36×, and 2 _._ 37×, respectively. Against Taylor series softmax, Mugi achieves 10 _._ 02×, 32 _._ 93×, and 3 _._ 28× improvements in the same metrics. 

Mugi contributes these gains in performance to its ability to scale to larger array sizes, sharing the compute array with GEMM. However, other designs require standalone vector arrays for nonlinear operations, where the scale is bounded by the SRAM bandwidth. Additionally, Mugi does not have to compute costly exp [68], thus multiplier free. These improvements allow Mugi to increase throughput and efficiency compared to vector arrays for both precise and approximation configurations. 

**Takeaway.** Mugi enables accurate nonlinear approximation by applying input approximation and value-centric approach to VLP, greatly enhancing both throughput and efficiency. 

## **6.2 GEMM** 

We show the GEMM execution results in Figure 12. The GEMM operations include the projection, attention, and FFN layers from the studied LLM models. We observe that in terms of throughput and efficiency, Mugi consistently outperforms both systolic and SIMD arrays. Mugi is optimized 

1225 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**Table 3.** Comparison of Mugi and baselines throughput, area, energy and power efficiency on LLaMa 2 70B (with GQA). Batch size is set to 8, and sequence length is set to 4096. Hardware details are outlined in Table 2. 

|**Design**|**Design**|**Throughput**<br>**(Tokens/s)**|**OC Area**<br>**(**_𝑚𝑚_2**)**|**Energy Efciency**<br>**(Tokens/s/**_𝜇_**J)**|**Power Efciency**<br>**(Tokens/s/W)**|
|---|---|---|---|---|---|
|SN|Mugi(128)<br>Mugi(256)<br>Carat (128)<br>Carat (256)|0.71<br>1.39<br>0.70<br>1.38|2.16<br>3.10<br>2.42<br>3.84|68.64<br>142.82<br>53.00<br>95.78|3.18<br>3.37<br>2.47<br>2.27|
||SA (16)|0.67|2.58|45.97|2.24|
||SA-F (16)<br>SD (16)<br>SD-F (16)|0.67<br>0.67<br>0.67|2.81<br>2.54<br>2.77|44.34<br>47.83<br>46.06|2.16<br>2.34<br>2.25|
|SN-S<br>SA (64)<br>2.70<br>25.84<br>138.59<br>1.68<br>SA-F (64)<br>2.70<br>29.56<br>131.66<br>1.60<br>SD (64)<br>2.70<br>25.14<br>143.18<br>1.74<br>SD-F (64)<br>2.70<br>28.86<br>135.79<br>1.65<br>Tensor<br>10.06<br>38.75<br>488.31<br>1.59||||||
|NoC|4 x 4Mugi(256)<br>|22.19|50.12|2314.23|3.42|
||4 x 4 Carat (256)<br>4 x 4 SA (16)<br>4 x 4 SA-F (16)<br>4 x 4 SD (16)<br>4 x 4 SD-F (16)<br>2 x 1 Tensor (8)|22.08<br>10.74<br>10.74<br>10.74<br>10.74<br>20.12|61.92<br>41.77<br>45.48<br>41.18<br>44.89<br>77.56|1551.09<br>770.31<br>741.68<br>803.82<br>772.70<br>989.02|2.30<br>2.35<br>2.26<br>2.45<br>2.36<br>1.61|



for 8 columns, which aligns with and benefits from a batch size and GQA group size of 8, allowing it to leverage GQA for better throughput. This capability ensures that Mugi maintains excellent utilization even as the array size scales. On the contrary, systolic and SIMD arrays face under-utilization with array sizes larger than 8x8. Additionally, VLP eliminates multiplication in Mugi, increasing efficiency compared to baselines. Compared with Carat [46], Mugi shows better efficiency as Carat needs additional hardware to execute nonlinear operations. 

**Takeaway** Mugi achieves good throughput and efficiency gains through timely LLM optimizations in emerging asymmetric data formats, small-batch GEMM, and GQA. 

## **6.3 LLM workload** 

**6.3.1 Single Node.** Single-node evaluations show that Mugi exceeds baseline implementations in both throughput and efficiency, while also reducing overhead and area costs. An end-to-end comparison is provided in Table 3. Compared to a baseline systolic array 16, the Mugi 256 architecture achieves an increase of 2 _._ 07×, 3 _._ 11×, and 1 _._ 50× in throughput, energy efficiency, and power efficiency respectively. These improvements stem from Mugi ’s effective reuse of the compute array for both nonlinear operations and GEMM, and efficient handling of asymmetric, small-batch GEMM using WOQ, KVQ, and GQA. 

Figure 13 highlights these advantages, which are further amplified by the elimination of specialized vector arrays and costly MAC units, resulting in a more compact compute array and lower power consumption. Mugi additionally scales more efficiently to larger array sizes, growing linearly. On the contrary, the area of systolic and SIMD arrays scales up 

**==> picture [242 x 309] intentionally omitted <==**

**----- Start of picture text -----**<br>
0% 50% 100%<br>Node: Fifo Nonlinear TC<br>Acc PE Vector VR<br>0.5 mm²<br>117.4 mW 128<br>35.1 mm²<br>2.6 W<br>Mugi<br>0.9 mm²<br>195.8 mW 256<br>50.1 mm²<br>4.1 W<br>0.8 mm²<br>122.1 mW 128<br>39.5 mm²<br>2.6 W<br>Mugi-L<br>1.5 mm²<br>205.3 mW 256<br>58.9 mm²<br>4.3 W<br>0.9 mm²<br>168.3 mW 128<br>39.2 mm²<br>3.2 W<br>Carat<br>2.0 mm²<br>325.0 mW 256<br>61.9 mm²<br>6.0 W<br>0.3 mm²<br>108.0 mW 8<br>28.5 mm²<br>2.0 W<br>S-F<br>1.0 mm²<br>221.6 mW 16<br>45.5 mm²<br>3.4 W<br>0.2 mm²<br>98.9 mW 8<br>28.4 mm²<br>1.9 W<br>SD-F<br>1.0 mm²<br>209.7 mW 16<br>44.9 mm²<br>3.3 W<br>NoC (4x4): Array SRAM NoC<br>**----- End of picture text -----**<br>


**Figure 13.** Array and NoC level area and power breakdown. Total area and power is shown in each bar. Cool colored bars represent array level while warm colored bars represent NoC level. _Acc_ refers to output accumulators, and _nonlinear_ refers to nonlinear hardware. For the power breakdown, batch size is set to 8 and the sequence length is set to 4096. 

quadratically as the array scales in both row and column dimensions. Though also based on VLP, Carat area scales up super-linearly, due to the excessive cost of FIFO. Despite the large area, the temporal coding in VLP still minimizes the switching activities and ensures a low power-to-area ratio. Comparing to Mugi, Mugi-L with LUT for nonlinear operations and VLP for GEMM, spends way more hardware on on-chip LUT, implemented using FIFOs to ensure programmability. 

We further show an extended throughput and energy comparison of different designs when sweeping the batch size, as shown in Figure 14, offering higher throughput and lower energy per token. The best throughput of Mugi is attainable at a smaller batch size of 8 than other baselines, as Mugi maps the batch across columns and peaks the utilization at a batch size of 8. 

We additionally considered off-chip memory accesses, and we see that Mugi handles DRAM traffic similarly to other 

1226 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

**==> picture [504 x 161] intentionally omitted <==**

**----- Start of picture text -----**<br>
Mugi (64) Carat (64) SA (8) SA-F (8) SD (8) SD-F (8)<br>Mugi (256) Carat (256) SA (16) SA-F (16) SD (16) SD-F (16)<br>Norm Throughput<br>128 256 512 1024 2048 4096<br>30x<br>20x<br>10x<br>Norm Energy/Token<br>25x<br>15x<br>5x<br>1 2 4 8 16 32 1 2 4 8 16 32 1 2 4 8 16 32 1 2 4 8 16 32 1 2 4 8 16 32 1 2 4 8 16 32<br>**----- End of picture text -----**<br>


**Figure 14.** ISO-throughput LLMs study, geometric mean over all Llama models. The x-axis represents batch size at the bottom and each plot represents sequence lengths (128-4096) at the top. The y-axis represents the improvements of throughput and energy per token. All designs are normalized to an 8×8 systolic array with batch a size of 1. -F denotes FIGNA, and array heights are indicated in “()”. SA and SD base and -F array’s throughput closely overlap, as do Mugi and Carat. 

dataflow architectures, almost identical operational intensity, but offers higher compute utilization, therefore more compute bounded. 

Figure 16 shows a latency breakdown of different LLMs. It is observed that Mugi have slightly better latency for attention GEMMs than other designs, but almost halves the latency for projection and FFN GEMMs. Regarding the nonlinear operations, Mugi shows almost invisible latency, exhibiting tremendous improvements over other designs. Though not obvious, Carat triples the nonlinear latency of Mugi, due to relying on non-VLP approximations. 

**6.3.2 Carbon Emission.** When comparing Mugi’s carbon impact to baselines, we see that Mugi improves in both operational and embodied emissions by 1 _._ 45× and 1 _._ 48× respectively. While Figure 15 shows operational carbon as the major contributor to emissions, this follows previous trends consitent with 45nm technologies. Through Mugi’s efficient, shared compute array, Mugi is able to simultaneously decrease both operational and embodied carbon while delivering an increase to throughput for LLM workloads. 

**6.3.3 Multi Node.** Mugi scales efficiently to multi-node designs using a NoC architecture. We compare Mugi ’s multinode implementations against baseline designs with the same NoC configuration and scaled-out versions of single node baseline architectures. As with single node setups, Mugi ’s multi node configurations show comparable gains in throughput, energy efficiency, and power efficiency when comparing Mugi 4x4 256 and systolic array 4x4 16. Figure 17 details these improvements, emphasizing the benefits of multi-node architectures. Moreover, NoC-based implementations clearly outperform scaled-up systolic arrays, due to severe under-utilization at a small batch size, as shown in Table 3. Figure 13 shows the breakdown of NoC level area, 

**==> picture [241 x 90] intentionally omitted <==**

**----- Start of picture text -----**<br>
Projection Attention FFN Nonlinear Embodied<br>7B 13B 70B 70B GQA<br>256 16 256 16 256 16 256 16<br>1.5<br>1.0<br>0.5<br>0.0<br>M C S D T P M C S D T P M C S D T P M C S D T P<br>**----- End of picture text -----**<br>


**Figure 15.** Normalized onchip operational and embodied carbon across model sizes (Llama 2-7B, 13B, 70B, 70B GQA) of Mugi and baseline configurations. Batch size is set to 8, and sequence length is set to 4096. Top x-axis represents array height, while y axis is normalized latency. M, C, S, D, T, P represents Mugi, Carat, Systolic, SIMD, Taylor Series, and Piecewise linear approximate respectively. 

**==> picture [242 x 95] intentionally omitted <==**

**----- Start of picture text -----**<br>
Projection Attention FFN Nonlinear<br>7B 13B 70B 70B GQA<br>256 16 256 16 256 16 256 16<br>2<br>1<br>0<br>M C S T P M C S T P M C S T P M C S T P<br>**----- End of picture text -----**<br>


**Figure 16.** Normalized end-to-end latency breakdown across model sizes. All notations are identical to those in Figure 15, except that S here is for Systolic/SIMD. 

and the array is occupying varying ratios of the on-chip resource, given an identical on-chip SRAM size. 

1227 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**==> picture [241 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
Mugi Carat SA SA-F SD SD-F Tensor<br>Norm Throughput<br>18x<br>12x<br>6x<br>Norm Energy Efficiency<br>30x<br>20x<br>10x<br>Norm Power Efficiency<br>2x<br>1x<br>64/8/S 128/16/2 256/SU/4<br>**----- End of picture text -----**<br>


**Figure 17.** Normalized NoC-level throughput, energy efficiency, and power efficiency for Mugi and baseline 4x4 and 8x8 NoC architectures. Data represents the geometric mean across model sizes (Llama 2–7B, 13B, 70B) with a batch size of 8 and sequence length set to 4096. The x-axis represents array height for each group, in order of VLP, SA/SD, and tensor core. Tensor core configurations represent -S (single node), 2 (2x1 NoC), 4 (2x2 NoC). All models are normalized to an 8×8 systolic array (NoC dim: 4x4). 

**Takeaway** By enabling VLP for both nonlinear operations and GEMM, Mugi effectively accelerates all aspects of LLM workloads, which also scales to multi nodes. 

## **7 Discussion** 

## **7.1 Limitations** 

While we showcase the benefits of VLP, there are a few workloads or techniques not addressed in this paper. 

**Additional Operations.** While Mugi unlocks nonlinear approximation, there are still some nonlinear LLM operations not supported, such as layer normalization and rotary positional embeddings (RoPE). Layer normalization are vector multiplication, and can be supported in Mugi’s vector unit. For RoPE, Mugi can either approximate the required sine and cosine functions, though the utilization might be low due to its sparse nature, or offload them to external hardware as in existing GEMM accelerators. 

**MoE and Multi-Modal Models.** Mixture-of-Experts (MoE) extend standard attention-based LLMs with selective FFN experts, selected by a softmax-based gating network [11, 15, 37]. Multi-modal models support multiple modalities beyond just text (i.e., language, vision, video, etc), by either tokenizing non-language inputs [14], or combining multiple modalityspecific layers [4, 62]. There additionally exist models that leverage MoE architecture on multiple modalities [40]. All these additional operations have been supported in Mugi, and different modality has been studied in this work to prove the efficacy of VLP. We conjecture Mugi should generalize 

to both MoE and multi-modal variants, though we leave full validation to future work. 

**Online Approximation.** Currently, Mugi pre-computes the results offline for accurate nonlinear approximation via LUT. However, the value distribution could exhibit a slight drift at runtime. Such drifts in both KV cache and FFN have been well tackled via quantization, using as few as 4 bits. As for softmax, since all softmax inputs are subtracted by the maximum for numerical stability, the drift minimally impacts accuracy. Moreover, Mugi ’s sliding window mechanism helps adapt to the current workload and further reduces the impact of drift. That said, we argue optimal accuracy would benefit from an online mechanism to adjust LUT values at runtime, and we leave this to future work. 

## **7.2 Related Works** 

**Nonlinear Approximation.** Prior works have explored approximating nonlinear operations. Some approaches use piece-wise linear (PWL) approximations, partitioning the function into segments and selecting coefficients based on input range, while others approximate the entire function with a single simplified equation [27, 67, 76, 79]. Other works use Taylor-series expansions, which can provide high accuracy but degrade as values drift from the expansion point [42, 66]. While all approximation techniques, including Mugi, introduce some levels of approximation error, others underperform Mugi in most models while increasing area and efficiency costs, as detailed in Figure 6 and Table 3. 

**GEMM Acceleration.** A number of prior works target GEMM acceleration. Carat first introduced VLP, enabling low-precision, large-batch value reuse for CNNs [46]. While large batch sizes compliment CNN workloads, LLMs operate with smaller batch sizes and larger matrices, making Carat unsuited for such workloads. Other accelerators exploit unary computing [73–75, 78] to reduce buffer overhead using ternary RIM arrays and unary half-adders [21], improving accumulation efficiency. However, each PE still requires multipliers and accumulators, whereas Mugi eliminates costly PE MAC units altogether via VLP. Additional works identify the ability to exploit data reuse to reduce data movement. Multi-chip module designs share inputs across weight-stationary chiplets to exploit cross-chiplet reuse [58]. Matrix-scaling approaches reduce large matrices into sub-matrices and scaling vectors, reducing memory transfers sharing sub-matrix computation during rescaling [53]. Lastly, another work identifies that quantization reduces the number of possible inputs, enabling computation to be shared between layers where outputs do not change [57]. Like Mugi, these works similarly identify data reuse techniques but largely overlook value reuse, missing further opportunities for optimization. 

**KV Cache Compatibility.** Some works employ detailed hardware–software co-design to improve LLM performance. 

1228 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

These works aim to reduce computation through top-k speculative prediction, removing computation predicted to be negligible to the attention output and reducing KV cache accesses [24, 52, 65, 66]. In contrast, other accelerators focus purely on GEMM computation and do not address attention or KV-cache related bottlenecks [21, 46, 53, 57, 58]. Mugi occupies a middle ground by incorporating lightweight KVcache optimizations natively in its architecture, avoiding workload-specific optimizations via co-design. This allows Mugi to generalize across models while still capturing key reuse opportunities in modern LLM workloads. 

## **8 Conclusion** 

In this paper, we orchestrate value-level parallelism (VLP) for efficient transformer-based LLMs. We formulate VLP for nonlinear approximation in a value-centric approach where important values are assigned with greater accuracy. We design a Mugi architecture for our VLP approximation. Additionally, we optimize Mugi to accelerate asymmetric, small-batch GEMM, which leverages the trending LLM optimizations. To this end, Mugi effectively supports full LLM workloads via VLP. Our experimental results demonstrate significant performance and efficiency gains in Mugi, highlighting the potential of VLP for AI workloads. 

1229 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

## **A Artifact Appendix** 

## **A.1 Abstract** 

The artifact evaluation is separated into two scopes, workload evaluation and architecture evaluation. The workload evaluation contains nearly all figures within section 3, namely Figure 4, Figure 6, and Figure 8. The architecture evaluation includes all figures within section 6. Both artifacts can be run on an x86_64 machine with python and conda installed. The workload evaluation requires access to a GPU cluster. We have tested the architecture evaluation on ubuntu 24.04, and the workload evaluation on an NSF GPU cluster running Red Hat 9.4. To run each artifact, download the artifacts from the zenodo links detailed in Section A.3 and follow steps outlined in Section A.4. To see the results generated from running the artifact, see Section A.6 for detail. 

## **A.2 Artifact check-list (meta-information)** 

**– Time needed to complete experiments (approximate):** 0.5 - 1 hours 

**– Publicly available:** Yes 

- **Code licenses (if publicly available):** MIT License 

- **Workflow framework used:** In-house simulation framework 

## **A.3 Description** 

**A.3.1 How to access.** Both artifacts can be downloaded at https://zenodo.org/records/18063514. Follow the instructions detailed in the zenodo or Section A.4 and A.5 to evaluate each artifact. 

**A.3.2 Hardware dependencies.** For the workload evaluation, a GPU capable of loading all models at half precision is required. 

**A.3.3 Software dependencies.** Conda is required to build the environment.[1] 

- **Workload evaluation:** 

- **Model:** AI profiling 

- **Data set:** Models outlined in Table 1. Note that Llama 70B results are excluded in the provided artifact, due to the excessive profiling time. 

- **Run-time environment:** Conda 

- **Hardware:** NSF GPU cluster 

- **Metrics:** Perplexity, value distribution, theoretical error 

- **Output:** Figure 6, Figure 4, Figure 8, 

- **Experiments:** Approximate perplexity comparison, model value distribution, approximation theoretical error 

- **Disk space required (approximate):** 70GB 

- **Time needed to complete experiments (approximate):** 12-24 hours 

- **Publicly available:** Yes 

- **Code licenses (if publicly available):** MIT License 

- **– Workflow framework used:** Pytorch 

- **Architecture evaluation:** 

- **Model:** Cycle-level performance model, event-based cost model 

- **Data set:** Llama configurations outlined in Table 1 

- **Run-time environment:** Conda 

- **Hardware:** x86_64 machine 

- **Metrics:** Throughput, latency, energy efficiency, power efficiency, area, carbon equivalent emissions. 

- **Output:** Figure 11, Figure 12, Table 3, Figure 13, Figure 14, Figure 15, Figure 16, Figure 17. 

- **Experiments:** Iso-area nonlinear comparison, isoarea GEMM comparison, comprehensive design comparison, array and noc-level area and power breakdown, batch size comparison, operational and embodied carbon comparison, end-to-end latency comparison, iso-area noc comparison. 

- **Disk space required (approximate):** 8GB 

**A.3.4 Datasets.** Both evaluations use the models outlined in Table 1, with the architecture evaluation only including Llama models, and the workload evaluation ignoring Llama 2 70b. Access to the ML models is provided in the artifact during the artifact evaluation, but will be deprecated after evaluation, and access to the models will have to be obtained.[2] 

**A.3.5 Models.** The models used in our simulation framework include a cycle-level performance model and an eventbased cost model, as well as profiling and end-to-end perplexity results. 

## **A.4 Installation** 

One may follow the steps below to run the artifact, also available at the zenodo link in Section A.3.1. For the artifact evaluation, we will provide access to the NSF GPU cluster. Both evaluations detail the steps assuming a Linux environment, and require the command-line tool unzip to be installed prior to evaluation. If you are evaluating either artifact on a local machine, follow the commands below to install unzip. Otherwise, ensure that unzip or and equivalent command-line tool is available. 

sudo apt update 

sudo apt install unzip 

## **Workload evaluation.** 

   1. Download the zip file from the zenodo link. mugi_profiling-asplos_2026_ae.zip https://zenodo.org/records/18063514 

   2. Unzip the artifact and cd into the new directory. unzip mugi_profiling-asplos_2026_ae.zip cd mugi_profiling-asplos_2026_ae 

- 1Available at https://www.anaconda.com/download. 

- 2Available at https://huggingface.co/meta-llama. 

1230 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

3. Create a conda environment with the included environment.yaml and activate the environment. conda env create -f environment.yaml conda activate mugi_profiling 

4. Run the included script to launch all slurm scripts. bash mugi_profiling.sh If slurm access is not available (NSF access fails), an additional script is provided, but tuning may be required. bash mugi_profiling_local.sh 

5. Finally, retrieve the figures in the figures/output directory. 

## **A.6 Evaluation and expected results** 

After running the steps in Section A.4, the generated figures can be found locally in figures/output directory for workload evaluation and zoo/llm/results/figs and zoo/llm/results/tables for the architecture evaluation. Each figure is labeled as figX-Y.pdf that corresponds to what is included in Section 3 and Section 6. The expected profiling distribution in the workload evaluation may exhibit slight deviations from the values reported in the paper due to device-specific computational variation. 

## **A.7 Methodology** 

Submission, reviewing and badging methodology: 

## **Architecture evaluation.** 

1. Download the zip file from the zenodo link. archx-asplos_2026_ae.zip https://zenodo.org/records/18063514 

   - https://www.acm.org/publications/policies/artifact-reviewbadging 

   - http://cTuning.org/ae/submission-20201122.html 

   - http://cTuning.org/ae/reviewing-20201122.html 

2. Unzip the artifact and cd into the new directory. unzip archx-asplos_2026_ae.zip cd archx-asplos_2026_ae 

3. Create a conda environment with the included environment.yaml and activate the environment. conda env create -f environment.yaml conda activate archx 

4. Run the included script to generate and simulate the architecture descriptions. bash run_mugi.sh 

5. Finally, retrieve the figures in the zoo/llm/results/figs/ and zoo/llm/results/tables/ directories. 

## **A.5 Experiment workflow** 

**Workload evaluation.** We use slurm scripts to automate profiling and end-to-end perplexity results. To produce each figure, the scripts load the model onto an allocated node and run the target dataset. Profiling is done on the base halfprecision models, while perplexity is retrieved for base models at both half precision and for nonlinear approximation. After all models run, the scripts process both the profiling and perplexity results and report them within each figure. 

**Architecture evaluation.** We use scripts to automatically run the workflow for the result production. To produce a figure, the evaluation scripts first generate all the hardware configurations based on the provided architecture descriptions. Then, the simulation is run on each architecture description with a target workload (e.g., llama). More specifically, both the performance model and cost model are run. Note that the automated workflow launches these simulations in parallel. Finally, the scripts parse the generated result and aggregate results across runs from all the configurations to generate figures. 

1231 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

## **References** 

- [1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. _USENIX Symposium on Operating Systems Design and Implementation_ (2024). 

- [2] Ankur Agrawal, Silvia M. Mueller, Bruce M. Fleischer, Xiao Sun, Naigang Wang, Jungwook Choi, and Kailash Gopalakrishnan. 2019. DLFloat: A 16-b Floating Point Format Designed for Deep Learning Training and Inference. In _2019 IEEE 26th Symposium on Computer Arithmetic (ARITH)_ . 

- [3] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. _Empirical Methods in Natural Language Processing_ . 

- [4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. 2025. Qwen3-VL Technical Report. _arXiv_ (2025). 

- [5] Rajeev Balasubramonian, Andrew B. Kahng, Naveen Muralimanohar, Ali Shafiee, and Vaishnav Srinivas. 2017. CACTI 7: New Tools for Interconnect Exploration in Innovative Off-Chip Memories. _Transactions on Architecture and Code Optimization_ (2017). 

- [6] Jeff Barr. 2019. Amazon EC2 Update – Inf1 Instances with AWS Inferentia Chips for High Performance Cost-Effective Inferencing. https://aws.amazon.com/blogs/aws/amazon-ec2-update-inf1instances-with-aws-inferentia-chips-for-high-performance-costeffective-inferencing/ 

- [7] Erik Brunvand, Donald Kline, and Alex K. Jones. 2018. Dark Silicon Considered Harmful: A Case for Truly Green Computing. In _nternational Green and Sustainable Computing Conference (IGSC)_ . 

- [8] Yuji Chai, John Gkountouras, Glenn G. Ko, David Brooks, and Gu-Yeon Wei. 2023. INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation. _arXiv_ (2023). 

- [9] Jaewan Choi, Hailong Li, Byeongho Kim, Seunghwan Hwang, and Jung Ho Ahn. 2022. Accelerating Transformer Networks through Recomposing Softmax Layers. In _International Symposium on Workload Characterization_ . 

- [10] Yujeong Choi, Yunseong Kim, and Minsoo Rhu. 2021. Lazy Batching: An SLA-aware Batching System for Cloud Machine Learning Inference. In _International Symposium on High-Performance Computer Architecture_ . 

- [11] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. _arXiv_ (2024). 

- [12] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, et al. 2025. Deepseek-V3 Technical Report. _arXiv_ (2025). 

- [13] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. In _Advances in Neural Information Processing Systems_ . 

- [14] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan 

Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. 2023. PaLM-E: an embodied multimodal language model. In _Proceedings of the 40th International Conference on Machine Learning_ . JMLR.org. 

- [15] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. 2022. GLaM: Efficient Scaling of Language Models with Mixture-of-Experts. In _Proceedings of the 39th International Conference on Machine Learning_ . 

- [16] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al. 2024. The Llama 3 Herd of Models. _arXiv_ (2024). 

- [17] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. 2018. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning. _Neural Networks_ (2018). 

- [18] Ahmad Faiz, Sotaro Kaneda, Ruhan Wang, Rita Osi, Prateek Sharma, Fan Chen, and Lei Jiang. 2024. LLMCarbon: Modeling the end-to-end Carbon Footprint of Large Language Models. _arXiv_ (2024). 

- [19] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. _arXiv_ (2023). 

- [20] Kunihiko Fukushima. 1969. Visual Feature Extraction by A Multilayered Network of Analog Threshold Elements. _IEEE Transactions on Systems Science and Cybernetics_ (1969). 

- [21] Hongrui Guo, Yongwei Zhao, Zhangmai Li, Yifan Hao, Chang Liu, Xinkai Song, Xiaqing Li, Zidong Du, Rui Zhang, Qi Guo, Tianshi Chen, and Zhiwei Xu. 2023. Cambricon-U: A Systolic Random Increment Memory Architecture for Unary Computing. In _International Symposium on Microarchitecture_ . 

- [22] Udit Gupta, Mariam Elgamal, Gage Hills, Gu-Yeon Wei, Hsien-Hsin S. Lee, David Brooks, and Carole-Jean Wu. 2022. ACT: designing sustainable computer systems with an architectural carbon modeling tool. In _International Symposium on Computer Architecture_ . 

- [23] Udit Gupta, Mariam Elgamal, Gage Hills, Gu-Yeon Wei, Hsien-Hsin S. Lee, David Brooks, and Carole-Jean Wu. 2022. ACT: designing sustainable computer systems with an architectural carbon modeling tool. 

- [24] Tae Jun Ham, Yejin Lee, Seong Hoon Seo, Soosung Kim, Hyunji Choi, Sung Jun Jung, and Jae W. Lee. 2021. ELSA: hardware-software codesign for efficient, lightweight self-attention mechanism in neural networks. In _International Symposium on Computer Architecture_ . 

- [25] Dan Hendrycks and Kevin Gimpel. 2016. Gaussian Error Linear Units (GELUs). _arXiv_ (2016). 

- [26] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. 2024. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. _arXiv_ (2024). 

- [27] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. 2019. Searching for MobileNetV3. _International Conference on Computer Vision_ (2019). 

- [28] Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, and Xiaojuan Qi. 2024. BiLLM: Pushing the Limit of Post-Training Quantization for LLMs. _arXiv_ (2024). 

- [29] International Telecommunication Union (ITU) and World Benchmarking Alliance (WBA). 2025. Tech sector emissions, energy use grow with rise of AI. https://www.itu.int/en/mediacentre/Pages/PR-202506-05-greening-digital-companies-report.aspx Accessed: 2025-08-12. 

1232 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Daniel Price, Prabhu Vellaisamy, John P. Shen, and Di Wu 

- [30] Jaeyong Jang, Yulhwa Kim, Juheun Lee, and Jae-Joon Kim. 2024. FIGNA: Integer Unit-Based Accelerator Design for FP-INT GEMM Preserving Numerical Accuracy. In _2024 IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ . 

- [31] Yangqing Jia. 2014. _Learning Semantic Image Representations at a Large Scale_ . Ph. D. Dissertation. EECS Department, University of California, Berkeley. http://www2.eecs.berkeley.edu/Pubs/TechRpts/2014/EECS2014-93.html 

- [32] Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. 2019. A Study of BFLOAT16 For Deep Learning Training. _arXiv_ (2019). 

- [33] Tushar and Qingru Zhang Kang, Hao, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao. 2024. GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM. _arXiv_ (2024). 

- [34] Rachid Karami, Sheng-Chun Kao, and Hyoukjun Kwon. 2025. NonGEMM Bench: Understanding the Performance Horizon of the Latest ML Workloads with NonGEMM Workloads. In _International Symposium on Performance Analysis of Systems and Software_ . 

- [35] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. ImageNet Classification with Deep Convolutional Neural Networks. In _Advances in Neural Information Processing Systems_ , F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger (Eds.). 

- [36] Andrey Kuzmin, Mart Van Baalen, Yuwei Ren, Markus Nagel, Jorn Peters, and Tijmen Blankevoort. 2022. FP8 Quantization: The Power of the Exponent. In _Advances in Neural Information Processing Systems_ . 

- [37] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. _arXiv_ (2020). 

- [38] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Guangxuan Xiao, and Song Han. 2025. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. _GetMobile: Mobile Comp. and Comm._ (2025). 

- [39] Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, and Furu Wei. 2024. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. _arXiv_ (2024). 

- [40] Meta AI. 2025. Llama 4: Multimodal Intelligence. https://ai.meta.com/ blog/llama-4-multimodal-intelligence/. Accessed: 2025-12-05. 

- [41] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. 2022. FP8 Formats for Deep Learning. _arXiv_ (2022). 

- [42] Peter Nilsson, Ateeq Ur Rahman Shaik, Rakesh Gangarajaiah, and title = Hardware Implementation of the Exponential Function Using Taylor Series journal = Nordic Circuits and Systems Conference year = 2014 Hertz, Er. [n. d.]. ([n. d.]). 

- [43] NVIDIA. 2024. NVIDIA H100 Tensor Core GPU Architecture. Retrieved 2024-11-14 from https://resources.nvidia.com/en-us-hopperarchitecture/nvidia-h100-tensor-c 

- [44] NVIDIA. 2024. TensorRT-LLM. Retrieved 2024-11-14 from https: //github.com/NVIDIA/TensorRT-LLM 

- [45] Stuart F. Oberman and Michael J. Flynn. 1997. Division Algorithms and Implementations. _IEEEXplore_ (1997). 

- [46] Zhewen Pan, Joshua San Miguel, and Di Wu. 2024. Carat: Unlocking Value-Level Parallelism for Multiplier-Free GEMMs. In _International Conference on Architectural Support for Programming Languages and Operating Systems_ . 

- [47] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreass Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. _arXiv_ (2019). 

- [48] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. 2021. Carbon Emissions and Large Neural Network Training. _arXiv_ (2021). 

- [49] Raspberry Pi. 2024. Raspberry Pi 5. Retrieved 2024-11-14 from https: //www.raspberrypi.com/products/raspberry-pi-5/ 

- [50] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. Efficiently Scaling Transformer Inference. In _Proceedings of Machine Learning and Systems_ . 

- [51] Valentina Popescu, Abhinav Venigalla, Di Wu, and Robert Schreiber. 2021. Representation Range Needs for 16-Bit Neural Network Training. _arXiv_ (2021). 

- [52] Yubin Qin, Yang Wang, Dazheng Deng, Zhiren Zhao, Xiaolong Yang, Leibo Liu, Shaojun Wei, Yang Hu, and Shouyi Yin. 2023. FACT: FFNAttention Co-optimized Transformer Architecture with Eager Correlation Prediction. In _International Symposium on Computer Architecture_ . 

- [53] Yubin Qin, Yang Wang, Zhiren Zhao, Xiaolong Yang, Yang Zhou, Shaojun Wei, Yang Hu, and Shouyi Yin. 2025. MECLA: Memory-ComputeEfficient LLM Accelerator with Scaling Sub-matrix Partition. In _International Symposium on Computer Architecture_ . 

- [54] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust Speech Recognition via Large-Scale Weak Supervision . In _International Conference on Machine Learning_ . 

- [55] Mariam Rakka, Jinhao Li, Guohao Dai, Ahmed Eltawil, Mohammed E Fouda, and Fadi Kurdahi. 2025. SoftmAP: Software-Hardware Codesign for Integer-Only Softmax on Associative Processors. In _Design Automation and Test in Europe_ . 

- [56] Prajit Ramachandran, Barret Zoph, and Quoc V. Le. 2017. Searching for Activation Functions. _arXiv_ (2017). 

- [57] Marc Riera, Jose-Maria Arnau, and Antonio Gonzalez. 2018. Computation Reuse in DNNs by Exploiting Input Similarity. In _International Symposium on Computer Architecture_ . 

- [58] Yakun Sophia Shao, Jason Clemons, Rangharajan Venkatesan, Brian Zimmer, Matthew Fojtik, Nan Jiang, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, Stephen G. Tell, Yanqing Zhang, William J. Dally, Joel Emer, C. Thomas Gray, Brucek Khailany, and Stephen W. Keckler. 2019. Simba: Scaling Deep-Learning Inference with Multi-Chip-Module-Based Architecture. In _International Symposium on Microarchitecture_ . 14–27. 

- [59] Haichen Shen, Lequn Chen, Yuchen Jin, Liangyu Zhao, Bingyu Kong, Matthai Philipose, Arvind Krishnamurthy, and Ravi Sundaram. 2019. Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis. In _Symposium on Operating Systems Principles_ . 

- [60] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. In _International Conference on Machine Learning_ . 

- [61] Jacob R. Stevens, Rangharajan Venkatesan, Steve Dai, Brucek Khailany, and Anand Raghunathan. 2021. Softermax: Hardware/Software CoDesign of an Efficient Softmax for Transformers. In _Design Automation Conference_ . 

- [62] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, et al. 2025. Gemini: A Family of Highly Capable Multimodal Models. 

1233 

Mugi : Value Level Parallelism For Efficient LLMs 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

- [63] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswamim, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. LLAMA 2: Open Foundation and Fine-Tuned Chat Models. _arXiv_ (2023). 

- [64] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All You Need. In _Advances in Neural Information Processing Systems_ . 

- [65] Huizheng Wang, Zichuan Wang, Zhiheng Yue, Yousheng Long, Taiquan Wei, Jianxun Yang, Yang Wang, Chao Li, Shaojun Wei, Yang Hu, and Shouyi Yin. 2025. MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness. In _International Symposium on Microarchitecture_ . 

- [66] Hanrui Wang, Zhekai Zhang, and Song Han. 2021. SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning. _International Symposium on High-Performance Computer Architecture_ (2021). 

_Architecture_ . 

   - [75] Di Wu, Jingjie Li, Ruokai Yin, Hsuan Hsiao, Younghyun Kim, and Joshua San Miguel. 2021. uGEMM: Unary Computing for GEMM Applications. _IEEE Micro_ (2021). 

   - [76] Di Wu and Joshua San Miguel. 2019. In-Stream Stochastic Division and Square Root via Correlation. In _Design Automation Conference_ . doi:10.1145/3316781.3317844 

   - [77] Di Wu and Joshua San Miguel. 2021. When Dataflows Converge: Reconfigurable and Approximate Computing for Emerging Neural Networks. In _International Conference on Computer Design_ . doi:10. 1109/ICCD53106.2021.00014 

   - [78] Di Wu and Joshua San Miguel. 2022. uSystolic: Byte-Crawling Unary Systolic Array. In _International Symposium on High-Performance Computer Architecture_ . 

   - [79] Di Wu, Ruokai Yin, and Joshua San Miguel. 2021. In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing. _IEEE Design & Test_ (2021). doi:10.1109/MDAT.2021.3050716 

   - [80] Di Wu, Ruokai Yin, and Joshua San Miguel. 2021. Normalized Stability: A Cross-Level Design Metric for Early Termination in Stochastic Computing. In _Asia and South Pacific Design Automation Conference_ . 

   - [81] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based Generative Models. In _USENIX Symposium on Operating Systems Design and Implementation_ . 

   - [82] Tianyi Zhang, Jonah Yi, Zhaozhuo Xu, and Anshumali Shrivastava. 2024. KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization. _arXiv_ (2024). 

   - [83] Youpeng Zhao, Di Wu, and Jun Wang. 2024. ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching. In _International Symposium on Computer Architecture_ . 

- [67] Shuo Wang, Zhe Li, Caiwen Ding, Bo Yuan, Qinru Qiu, Yanzhi Wang, and Yun Liang. 2018. C-LSTM: Enabling Efficient LSTM using Structured Compression Techniques on FPGAs. _International Symposium on Field-Programmable Gate Arrays_ (2018). 

- [68] Maciej Wielgosz and Ernest Jamro. 2009. Highly Efficient Twin Module Structure of 64-Bit Exponential Function Implemented on SGI RASC Platform. _ResearchGate_ (2009). 

- [69] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. 2020. HuggingFace’s Transformers: State-of-the-art Natural Language Processing. _arXiv_ (2020). 

- [70] Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan Maeng, Gloria Chang, Fiona Aga Behram, James Huang, Charles Bai, Michael Gschwind, Anurag Gupta, Myle Ott, Anastasia Melnikov, Salvatore Candido, David Brooks, Geeta Chauhan, Benjamin Lee, Hsien-Hsin S. Lee, Bugra Akyildiz, Maximilian Balandat, Joe Spisak, Ravi Jain, Mike Rabbat, and Kim Hazelwood. 2022. Sustainable AI: Environmental Implications, Challenges and Opportunities. _arXiv_ (2022). 

- [71] Di Wu, Tianen Chen, Chienfu Chen, Oghenefego Ahia, Joshua San Miguel, Mikko Lipasti, and Younghyun Kim. 2019. SECO: A Scalable Accuracy Approximate Exponential Function Via Cross-Layer Optimization. In _International Symposium on Low Power Electronics and Design_ . doi:10.1109/ISLPED.2019.8824959 

- [72] Di Wu, Jingjie Li, Setareh Behrooz, Younghyun Kim, and Joshua San Miguel. 2021. UNO: Virtualizing and Unifying Nonlinear Operations for Emerging Neural Networks. In _International Symposium on Low Power Electronics and Design_ . 

- [73] Di Wu, Jingjie Li, Zhewen Pan, Younghyun Kim, and Joshua San Miguel. 2022. uBrain: A Unary Brain Computer Interface. In _International Symposium on Computer Architecture_ . 

- [74] Di Wu, Jingjie Li, Ruokai Yin, Hsuan Hsiao, Younghyun Kim, and Joshua San Miguel. 2020. uGEMM: Unary Computing Architecture for GEMM Applications. In _International Symposium on Computer_ 

1234 

