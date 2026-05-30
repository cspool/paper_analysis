# B. Computations in Large ML Models

Contemporary ML models contain embarrassingly parallel computations, such as, Generalized Matrix Multiplication (GeMM), 2-D Convolution (Conv2D), Dropout, and Softmax. We consider four widely used machine learning models: MegatronLM GPT-3 145B [12], LLaMA 65.2B [15], ResNet-38 [6], and VGG-19 [13]. Below we briefly explain computations involved in these models.

1) Transformers Models: A transformer is a deep learning architecture for Natural Language Tasks and is the basis of two widely used models: MegatronLM GPT-3 [12] and LLaMA [15].

```
1 //X: [B, S, H]; W<sub>1</sub>: [H, H/3];
2 //V: [H, H/3]; W<sub>2</sub>: [H/3, H]
3 //1st GeMM XW<sub>1</sub>: [B, S, H/3]
4 XW<sub>1</sub> = GeLU(X × W<sub>1</sub>)
5 //2nd GeMM XV: [B, S, H/3]
6 XV = X × V
7 //SwiGLU fused with 3rd GeMM XW<sub>2</sub>: [B, S, H]
8 SwiGLU = Swish(XW<sub>1</sub>) · XV
9 XW<sub>12</sub> = SwiGLU × W<sub>2</sub>
```

Fig. 3: The LLaMA MLP contains three weight matrices. With model parallelism on 8 GPUs, these matrices are:  $W_1$  of shape  $\left[\mathbb{H}, \frac{\mathbb{H}}{3}\right]$ , V of shape  $\left[\mathbb{H}, \frac{\mathbb{H}}{3}\right]$ , and  $W_2$  of shape  $\left[\frac{\mathbb{H}}{3}, \mathbb{H}\right]$ .

An inference request to a transformer model consists of a prompt and is served in two phases: (i) prompt processing, where the prompt is processed, and (ii) token generation, where a series of tokens that represents the output response text is generated incrementally. The model can batch B requests into a single inference task. The sequence length S denotes the number of tokens of each request being processed in the prompt processing phase or the number of tokens of each request being generated in the token generation phase. Therefore, during prompt processing  $B \ge 1$ , B > 1 and during token generation B > 1, B = 1.

A transformer consists of multiple Multi-Layer Perceptron (MLP) and Attention blocks. The design of MLP and Attention can be different for each model.

GPT-3: In GPT-3, both MLP and Attention takes an input matrix, perform operations with its two weight matrices, and outputs a matrix. With model parallelism these weight matrices are divided among all GPUs [12]. Figure 2 shows computations of GPT-3 with model parallelism of 8 GPUs. Both MLP and Attention first applies a linear transformation on the input, i.e., perform GeMM of the input and the weight matrix. Then, they perform operations, such as, GeLU and the Attention mechanism. Finally, the output of this operation is applied to second linear transformation. Existing MLP implementations fuse the GeLU activation with the first GeMM (line 4 in Figure 2a). State-of-the-art Attention implementations [4] caches the already processed and generated tokens in a KV Cache, such that, after prompt processing number of cached tokens, i.e. S', is set to S and when generating tokens S' increases incrementally. These implementations also fuses the attention mechanism in a single CUDA kernel (line 11-line 13) in Figure 2b.

**LLaMA**: LLaMA uses the hidden dimension of size 8192. LLaMA's MLP contains three GeMMs and SwiGLU [11] activation as shown in Figure 3. State-of-the-art implementations combines first two GeMMs into a single GeMM and fuses the SwiGLU activation with the third GeMM. Moreover, LLaMA uses the same Attention architecture as GPT-3.

2) Computer Vision Models: ResNet-38 [6] and VGG-19 [13] are two state-of-the-art computer vision models, where each layer performs several Conv2D operations. Table II shows

<span id="page-3-3"></span>TABLE II: INPUT/OUTPUT IMAGE SIZE (P, Q, C), KERNEL SIZE (R, S), CHANNELS (K) FOR EACH CONV2D, NUMBER OF CONV2DS PER LAYER, AND NUMBER OF LAYERS IN RESNET-38 AND VGG-19.

| [P, Q, C]     | [R, S] | K   | Convs/Layer   |   | Layers |     |
|---------------|--------|-----|---------------|---|--------|-----|
|               |        |     | ResNet<br>VGG |   | ResNet | VGG |
| [56, 56, 64]  | [3, 3] | 64  | 2             | 2 | 3      | 1   |
| [28, 28, 128] | [3, 3] | 128 | 2             | 2 | 4      | 1   |
| [14, 14, 256] | [3, 3] | 256 | 2             | 4 | 6      | 1   |
| [7, 7, 512]   | [3, 3] | 512 | 2             | 4 | 3      | 1   |

the details of each convolution layer.

