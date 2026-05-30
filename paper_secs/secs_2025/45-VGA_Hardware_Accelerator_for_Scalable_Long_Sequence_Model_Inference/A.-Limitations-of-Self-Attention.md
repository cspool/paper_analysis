# *A. Limitations of Self Attention*

Self-attention [48] is a powerful mechanism that dynamically captures the relationships between inputs. Self-attentionbased transformer models have achieved state-of-the-art performance in a wide range of domains, including natural language processing [9], [15], computer vision [16], and scientific domains such as protein 3D structure prediction [32]. For a length-l sequence of d-dimensional input, self-attention first passes the input through three separate fully connected layers to acquire 3 (l × d) dimensional matrices, respectively called Query (Q), Key (K), Value (V ) matrices. Then, Q and K are multiplied to acquire a (l × l) score matrix (S), the rows of which are normalized using the softmax function. Finally, S and V are multiplied to produce the (l × d) attention matrix.

Despite its groundbreaking success, modeling long sequences remains a challenge for self-attention. This is because the computation and memory cost of self-attention increases quadratically with the input sequence length l. Also, selfattention and its approximation variants tend to underperform on tasks where modeling of long-range context is important [47].

