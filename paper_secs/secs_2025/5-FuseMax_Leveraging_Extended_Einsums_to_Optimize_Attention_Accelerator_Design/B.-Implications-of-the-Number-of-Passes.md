# B. Implications of the Number of Passes

The number of passes a cascade performs is relevant because it restricts possible fusion schedules. Einsums within a pass can be fused at will, producing and consuming a tile of the intermediate at a time. Einsums in different passes cannot be fused. Revisiting Cascade 1, Einsums 5 and 6 cannot be fused on the K rank. Any implementation must visit all elements of the K fiber of K to produce K before it can visit any of the elements of that fiber to produce K.

This analysis also provides a non-trivial lower bound on the tensors' live footprints. For example, the algorithmic minimum live footprint for tensor A is a fiber of shape K. In other words, an architecture must either have enough buffer space to hold an entire K fiber of A or spill and reload that fiber, incurring memory traffic proportional to the shape of K. We note that this analysis is mapping independent. There is no dataflow for this cascade that enables a smaller live footprint.

