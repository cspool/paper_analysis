# II. BACKGROUND AND MOTIVATION

#### A. Quantization Schemes

Quantization is an effective technique to compress neural networks by reducing the bitwidth of parameters. According to the coding length, existing quantization scheme can be classified into fixed-length quantization and mixed-precision quantization, where fixed-length quantization can be further classified into uniform and non-uniform quantization.

![](_page_1_Figure_9.jpeg)

Fig. 1. Compression-based encoding comparison. Prior quantization works adopt (a) sparsity-based encoding that store significant and normal values separately; (b) quantization-based encoding that sacrifice the normal value to store the significant value. (c) Our proposed SPARK encoding stores significant and normal values together.

a) Uniform quantization: Uniform quantization [48] is the most widely used quantization method. For m-bit uniform scheme, it quantizes the data uniformly to  $2^m$  quantization levels:

$$Q^{FP}(m,\alpha) = \pm \alpha \times \left\{0, \frac{1}{2^{m-1}-1}, \frac{2}{2^{m-1}-1}, \dots, 1\right\}$$
 (1)

where  $\alpha$  is the scaling factor. Then for a 32-bit floating point value w, it is quantized to an m-bit quantized value  $\hat{w}$  by the following equation:

$$\hat{w} = \alpha \cdot h^{-1} \left( \frac{1}{2^m - 1} \operatorname{round} \left( (2^m - 1) \cdot h(\lceil w, \alpha \rfloor) \right) \right)$$
 (2)

where the function h(...) transforms a value into the range [0,1], and  $\lceil w,\alpha \rfloor$  scales w by  $\alpha$ . The quantized values are rounded to  $2^m$  fixed quantized values. This approach allows values to be quantized uniformly without excessive error, such as INT [16].

b) Non-uniform quantization: Non-uniform quantization does not distribute the quantized values uniformly, but rather distributes them more appropriately according to the characteristics of the data. A non-uniform quantization such as power of two quantization [54] [19], which quantizes the values as a sum of multiple power-of-two terms, allows for better fitting of Gaussian-like distributed weights. AdaptiveFloat [42] is another representative work, which performs a floating-point

TABLE I
COMPARISON BETWEEN PROPOSED METHOD SPARK AND SOME OTHER
EXISTING METHODS.

| Method   | Encoding                              | Memory Aligned                  | Data-free |
|----------|---------------------------------------|---------------------------------|-----------|
| OLAccel  | Coordinate list                       | ×                               | ✓         |
| GOBO     | Coordinate list<br>(Only for weights) | ×                               | ✓         |
| BiScaled | Block sparse index                    | Aligned data<br>Unaligned index | ✓         |
| ANT      | Mixed-type quantization               | ✓                               | ×         |
| SPARK    | Variable Length Encoding              | ✓                               | <b>√</b>  |

expansion based on the standard format of IEEE-754 to adaptively reduce quantization errors in Gaussian-like distribution.

c) Mixed-precision quantization: Mixed-precision quantization is a powerful technique that leverages different bitwidths or data types to represent values in Deep Neural Networks (DNNs) [45] [7]. This approach has been demonstrated to better preserve accuracy while using lower bit widths. The reason for this success lies in the fact that the parameters in DNN models exhibit varying importance and sensitiveness to the bit length. For instance, BitFusion [38], a neural network accelerator, implements mixed-precision quantization by allowing layer-wise changes in quantization accuracy through a combination of Multiply-Accumulate (MAC) units. Similarly, ANT [11] employs a mixture of integer, float, and power-oftwo data types for mixed quantization based on data range. However, this data type setup introduces retraining, which is resource-intensive and particularly demanding for complex attention-based models due to substantial memory requirements.

#### B. Compression-based Encoding

Various compression-based designs have emerged due to differences in parameter distributions, as shown in Table I. OLAccel [35] divides tensor values into two regions: outliers and normal values. Outliers are represented with high precision, while normal values are compressed with fewer bits. GOBO [50] employs a list of coordinates to indicate the locations of different data representations, using high-precision (8-bit or 16-bit) quantization for larger data representations. BiScaled DNN [17] quantizes all values with the same bitwidth but different scale factors for alignment, while ANT [11] uses encodings of the same bit-width but different data types for alignment, and utilizes finetuning to ensure precision. Olive [10] imposes a tight constraint on parameter distributions by finding significant values, pruning adjacent parameters, and utilizing the bit-width of pruned values to store larger values. However, these compression-based designs increase hardware complexity as a trade-off for their respective advantages.

