# C. Reducing the Number of Passes via Reassociation

Given the restrictions that multi-pass cascades place on the allowed dataflows and tensor live footprints, it can be beneficial to manipulate the cascade to reduce the number of passes required. Crucially, these manipulations are functionally equivalent and only change how Z is computed. In this section, we will present two methods for doing so, though we leave a full analysis of the space of pass-reduction approaches to future work.

1) Deferring the Multiplication by Y: First, we recognize that, by the distributive property, Einsum 6 can be factored to perform the reduction of  $A_k$  first, before multiplying the result by Y. Doing so, we get the following cascade:

$$Y = A_k \times B_k \tag{7}$$

$$X = A_k \tag{8}$$

$$Z = Y \times X \tag{9}$$

Cascade 2: A reassociation of Cascade 1 that defers the  $Y \times$  to compute Z with 1-pass of the K rank.

Now, because there is no read-after-write dependency between Einsums 7 and 8, both Einsums can be included in the same pass. In fact, because Einsum 8 reduces away the K rank, Cascade 2 is a 1-pass cascade with respect to this rank. This reassociation actually provides a second benefit over Cascade 1: Einsum 9 now only requires one multiplication (as opposed to K multiplications in Einsum 6).

2) Iteratively Constructing Y and Z: Alternatively, we can iteratively construct Y and Z as we perform the pass through  $A_k$ . To do so, we will take a similar approach to the prefix-sum (see Sections II-C3-II-C4) and build intermediate Ys and Zs.

$$RY_{i+1} = A_{k:k \le i} \times B_{k:k \le i} \tag{16}$$

$$RZ_{i+1} = RY_{i+1} \times A_{k:k \le i}$$
 (17)

Just like with the prefix sum, this version requires a lot of extra compute, but, because  $Y = RY_K$  and therefore  $Z = RZ_K$ , the final result is the same.

Initialization:

$$RY_{i:i=0} = 0$$
 (10)

$$RZ_{i\cdot i=0} = 0 \tag{11}$$

Extended Einsums:

$$RY_{i+1} = RY_i + A_i \times B_i \tag{12}$$

$$RZ_{i+1} = RZ_i \times \frac{RY_{i+1}}{RY_i} + RY_{i+1} \times A_i \qquad (13)$$

$$Z = RZ_K \tag{14}$$

$$\diamond: i > K \tag{15}$$

Cascade 3: A reassociation of Cascade 1 that iteratively constructs Y and Z with 1-pass of the K rank.

We remove this extra work by making the I ranks of  $RY_{i+1}$  and  $RZ_{i+1}$  iterative. This is shown in Cascade 3. Iterative  $RY_{i+1}$  (Einsum 12) looks very similar to the iterative prefixsum. However, computing  $RZ_{i+1}$  is a little more complicated.

To derive the expression for  $RZ_{i+1}$ , we start by introducing one more intermediate  $S_i$ , which is the prefix sum for  $A_k$ :

$$S_i = A_{k:k < i-1} (18)$$

Now, we can combine Einsums 17 and 18 to write  $RZ_i$  in terms of this prefix-sum:

$$RZ_i = RY_i \times S_i \tag{19}$$

Dividing both sides by  $RY_i$ , we derive an alternate definition for  $S_i$ :

$$S_i = \frac{RZ_i}{RY_i}$$

 $S_{i+1}$  can also be written using this alternative definition:

$$S_{i+1} = \frac{RZ_i}{RY_i} + A_i \tag{20}$$

We can combine Einsums 19 and 20 to compute  $RZ_{i+1}$  in terms of  $RZ_i$  (i.e., iteratively):

$$RZ_{i+1} = RY_{i+1} \times \left(\frac{RZ_i}{RY_i} + A_i\right)$$

Distributing  $RY_{i+1}$  and performing some reassociation, we get Einsum 13.

Cascade 3 is also a 1-pass cascade, performing one pass of the K rank of  $A_k$  (indexed with the variable i) and iteratively building  $RY_{i+1}$  and  $RZ_{i+1}$ . Unfortunately, unlike Cascade 2, Cascade 3 does require extra compute over the original Cascade 1. However, memory bandwidth-limited workloads can afford to trade off extra compute for reduced memory traffic, and Cascade 3 may still provide benefit.

![](_page_5_Figure_22.jpeg)

Fig. 1: Overview of transformer encoder inference.

## IV. TAXONOMIZING ATTENTION AS EINSUM CASCADES

Our second contribution is to apply the cascade of Einsums abstraction and the notion of passes to transformer models to describe, taxonomize, and highlight trade-offs in the space of attention implementations. This section first looks at the transformer model as a whole, identifying attention as an important kernel (Section IV-A). We then give an overview of attention and a "straightforward" (but inefficient) algorithm for softmax by writing them as cascades of Einsums (Sections IV-B-IV-C). Finally, we show how optimizations to softmax can be described by modifying the cascades and provide a taxonomy of the space using the number of passes required by each cascade (Sections IV-D-IV-E).

### A. Transformers

Transformer models generally follow the architecture defined in [52]. Our work, which addresses the impact of long sequence lengths during self-attention, focuses on the encoder architecture. Figure 1a gives an overview. The transformer first projects the input (by multiplying it by weight tensors) to form a *query*, *key*, and *value*. Self-attention is made up of three operations: a matrix multiplication of the query and key, a softmax on the result, and another matrix multiplication, which combines the softmax output with the value. The attention output is then deprojected (again, multiplying by a weight tensor), normalized, passed through a two-layer feed-forward neural network (FFN), and normalized once more.

As the sequence length grows, the relative importance of the different operations changes. Figure 1b shows that at shorter sequence lengths, the *weight-times-activation* "linear" layers are a larger fraction of the total required compute, while at long sequence lengths, the attention operation dominates. In all cases, the additional non-linearities (e.g., the normalization, the ReLU between the FFN layers, etc.) have negligible impact. In the next section, we focus on describing attention more precisely, and use our analysis to understand prior work on efficient implementations.

<sup>1</sup>During the decoder phase, inference is severely bottlenecked on the memory traffic required to read the KV cache [24], and therefore the on-chip accelerator design has less impact on performance.

## B. Redefining Attention's "Matrix Multiplications"

In the original transformer paper [52], the kernel was described with the following equation:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
 (21)

However, this equation says almost nothing about what the inputs Q, K, and V look like or what iteration space needs to be traversed. We clarify these points by rewriting the above as a cascade of Einsums, with the exception of the softmax, whose cascade we will explore in Section IV-C. The first step is to give each of the ranks names: M and P are the sequence lengths for Q and K/V, respectively, and E and F are the embeddings for Q/K and V, respectively.

$$QK_{m,p} = \frac{1}{\sqrt{E}} \times Q_{e,p} \times K_{e,m}$$
 (22)

$$A_{m,p} = softmax(QK_{m,p}) \tag{23}$$

$$AV_{f,p} = A_{m,p} \times V_{f,m} \tag{24}$$

Here, Einsums 22<sup>2,3</sup> and 24 look like matrix multiplications. Taking Einsum 24 as an example, for each point in the iteration space  $F \times M \times P$ , we perform a multiplication using elements from two 2-tensors  $(A_{m,p} \text{ and } V_{f,m})$  to produce a 2-tensor output  $(AV_{f,p})$ , which requires reducing across the inputs' shared rank M. Einsums 22-24 can be modified to refer to the full batched, multi-head self attention [52] by adding the batch (B) and head (H) ranks to all tensors. This changes the characteristics of the kernel. Adding the B and H ranks means that Einsums 22 and 24 behave like many independent matrix multiplications instead of one monolithic matrix multiplication. The challenges with attention, described in Section I, still follow clearly from this modification. Because all tensors contain a B rank, the matrix multiplications are all unique to the specific batch's inputs. Therefore, none of these tensors can be computed before the inputs are given, and there is no data sharing between the different elements in the batch. Hence, to simplify notation, we assume the presence of the B and Hranks but omit writing them throughout the rest of paper.

