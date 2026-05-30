# H Extension to Convolutional Layers

Low-rank adapters can also be applied to convolutional layers. Given an input feature map *X* ∈ <sup>R</sup>*h*×*w*×*c*<sup>1</sup> and *<sup>c</sup>*<sup>2</sup> 2D convolutional kernels *<sup>K</sup><sup>i</sup>* <sup>∈</sup> <sup>R</sup>*c*1×*d*×*<sup>d</sup> , i* = 1*,*2*,..., c*2, the output of the convolutional layer is

<span id="page-21-3"></span>
$$Y = \operatorname{stack}(X \otimes K_1, ..., X \otimes K_{c_2}), \tag{10}$$

where *Y* ∈ R*h*×*w*×*c*<sup>2</sup> and ⊗ denotes the 2D convolution operation.

<span id="page-22-0"></span>Table 18: Results of LoftQ using 2-bits uniform quantization compared with LoSparse with DeBERTaV3-base models on some of GLUE development sets. Here *Ratio* is the proportion of total remaining weights. Results with *N.A.* indicate the model does not converge.

| Method   | Ratio | MNLI<br>m / mm | SST-2<br>Acc | QNLI<br>Acc |  |
|----------|-------|----------------|--------------|-------------|--|
| Full FT  | 100%  | 90.5 / 90.6    | 95.3         | 94.0        |  |
| LoSparse | 15%   | 83.3/82.9      | 87.6         | 90.4        |  |
|          | 20%   | 84.5/83.8      | 91.7         | 88.6        |  |
| LoftQ    | 15.6% | 87.3/87.1      | 94.0         | 90.6        |  |
|          | 18.8% | 88.0/88.1      | 94.7         | 92.4        |  |

We can reformulate Equation [\(10\)](#page-21-3) into matrix multiplication as

$$Y = Z \times H^{\top},$$

where *Z* ∈ R*hw*×*c*1*<sup>d</sup>* 2 *,H* ∈ R*c*2×*c*1*<sup>d</sup>* 2 , by extending and flattening the input *X* together with concatenating and flattening kernels. We first extend a vector *xi,j* ∈ R*c*<sup>1</sup> by its neighbor vectors within the kernel window:

$$x_{i,j}^{'} = \text{Concat}(x_{i-\frac{d}{2},j-\frac{d}{2}},...,x_{i+\frac{d}{2},j+\frac{d}{2}}).$$

Now, *X* becomes *X* ′ ∈ R*h*×*w*×*c*1*<sup>d</sup>* . We then flatten *X* ′ into *Z* ∈ R*hw*×*c*1*<sup>d</sup>* 2 . For kernels, we first concatenate {*K*1*,...,Kc*<sup>2</sup> } into *H*′ ∈ R*c*2×*c*1×*d*×*<sup>d</sup>* . We then flatten *H*′ into *H*.

Note that *H* can be approximated by a low-rank matrix

$$R = UV^{\top},$$

where *U* ∈ R*c*2×*<sup>r</sup> ,V* ∈ R*c*1*<sup>d</sup>* <sup>2</sup>×*r , r* ≪ min{*c*2*, c*1*d* 2 } by SVD. Therefore, the original convolution layer can be approximated as

$$\widehat{Y} = Z \times (UV^{\top})^{\top} \tag{11}$$

$$= (Z \times V) \times U^{\top} \tag{12}$$

$$= M \times U^{\top}. \tag{13}$$

Note that *Z* × *V* can be restored into a convolution operation where we have *r* kernels *D<sup>i</sup>* ∈ R*c*1×*d*×*<sup>d</sup> , i* = 1*,*2*,,..., r* and *M* × *U*<sup>⊤</sup> can also be restored into a convolution operation where we have *<sup>c</sup>*<sup>2</sup> kernels *<sup>U</sup><sup>i</sup>* <sup>∈</sup> <sup>R</sup>*r*×1×<sup>1</sup> *, i* = 1*,*2*,,..., c*2.