# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

<span id="page-0-1"></span>Yu Sun∗<sup>1</sup> , Xinhao Li∗<sup>2</sup> , Karan Dalal∗<sup>3</sup> , Jiarui Xu<sup>2</sup> , Arjun Vikram<sup>1</sup> , Genghan Zhang<sup>1</sup> , Yann Dubois<sup>1</sup> , Xinlei Chen†<sup>4</sup> , Xiaolong Wang†<sup>2</sup> , Sanmi Koyejo†<sup>1</sup> , Tatsunori Hashimoto†<sup>1</sup> , Carlos Guestrin†<sup>1</sup> <sup>1</sup> Stanford University <sup>2</sup> UC San Diego <sup>3</sup> UC Berkeley <sup>4</sup> Meta AI

#### Abstract

Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their hidden states. We present a practical framework for instantiating sequence modeling layers with linear complexity and expressive hidden states. The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences, our layers are called *Test-Time Training (TTT) layers*. We consider two instantiations: TTT-Linear and TTT-MLP, whose hidden state is a linear model and a two-layer MLP respectively. We evaluate our instantiations at the scale of 125M to 1.3B parameters, comparing with a strong Transformer and Mamba, a modern RNN. Similar to Transformer, TTT-Linear and TTT-MLP can keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context. TTT-MLP still faces challenges in memory I/O, but shows larger potential in long context, pointing to a promising direction for future research.

<span id="page-0-0"></span>![](_page_0_Figure_5.jpeg)

Figure 1. All sequence modeling layers can be expressed as a hidden state that transitions according to an update rule. Our key idea is to make the hidden state itself a model *f* with weights *W* , and the update rule a gradient step on the self-supervised loss *ℓ*. Therefore, updating the hidden state on a test sequence is equivalent to training the model *f* at test time. This process, known as Test-Time Training (TTT), is programmed into our TTT layers.

<sup>∗</sup> Core contributors. † Joint advising. See author contributions at the end of the paper. Correspondence to: ys646@stanford.edu, xil202@ucsd.edu, kdalal@berkeley.edu. Code available in [JAX](https://github.com/test-time-training/ttt-lm-jax) and [PyTorch.](https://github.com/test-time-training/ttt-lm-pytorch)

The first version of this paper was submitted to arXiv on July 5, 2024. The current version contains updates on related work and limitations. All experiments were completed in the first version.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

![](_page_1_Figure_1.jpeg)

Figure 2. Comparing to Mamba, TTT-Linear and TTT-MLP have similar perplexity in 8k context (left) and better use of long context (right). Evaluations follow Kaplan et al. [\[43\]](#page-21-0). Left: Scaling trends on the Pile with 8k context, zoomed in between 350M and 1.3B parameters. Right: Similar to Transformer, TTT-Linear and TTT-MLP can keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context. All methods have matched training FLOPs as Mamba 1.4B.

## 1 Introduction

In 2020, the OpenAI scaling law paper (Kaplan et. al [\[43\]](#page-21-0)) showed that LSTMs (a type of RNN) could not scale similarly to Transformers or effectively use long context. Now, with modern RNNs and best practices, we re-evaluate these findings in Figure [2.](#page-1-0)

On the left, we observe that Mamba [\[27\]](#page-20-0) – one of the most popular RNNs today – scales similarly to a strong Transformer, showing great progress since the LSTMs in 2020. However, on the right, we observe the same issue with Mamba as Kaplan et al. did with LSTMs. Tokens later in a sequence should be easier to predict on average, since they condition on more information. This is indeed the case for Transformer, whose average perplexity at each token index decreases throughout its 32k context. In contrast, the same metric plateaus for Mamba after 16k.

This result represents an awkward reality for existing RNNs. On one hand, the main advantage of RNNs (vs. Transformers) is their linear (vs. quadratic) complexity. This asymptotic advantage is only realized in practice for long context, which according to Figure [12](#page-14-0) is after 8k. On the other hand, once context is long enough, existing RNNs such as Mamba struggle to actually take advantage of the extra information being conditioned on.

The difficulty with long context is inherent to the very nature of RNN layers: Unlike self-attention, RNN layers have to compress context into a hidden state of fixed size. As a compression heuristic, the update rule needs to discover the underlying structures and relationships among thousands or potentially millions of tokens. This need is inherently challenging. In this paper, we begin with the observation that self-supervised learning can compress a massive training set into the weights of a model such as an LLM, which often exhibits deep understanding about the semantic connections among its training data – exactly what we need from a compression heuristic.

TTT layers. Motivated by this observation, we make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences, these RNN layers are called *Test-Time Training (TTT) layers*. We introduce two simple instantiations: TTT-Linear and TTT-MLP, where the hidden state is a linear model and a two-layer MLP, respectively. TTT layers can be integrated into any network architecture and optimized end-to-end, similar to RNNs layers and self-attention.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

|                | Initial state       | Update rule                           | Output rule                                | Cost |
|----------------|---------------------|---------------------------------------|--------------------------------------------|------|
| Naive RNN      | = vector()<br>s0    | = σ (θssst−1<br>+ θsxxt<br>)<br>st    | = θzsst<br>+ θzxxt<br>zt                   | O(1) |
| Self-attention | s0<br>= list()      | st<br>= st−1.append(kt<br>, vt<br>)   | = Vtsoftmax<br><br>T<br>zt<br>K<br>qt<br>t | O(t) |
| Naive TTT      | W0<br>= f .params() | −<br>η∇ℓ(Wt−1;xt<br>Wt<br>= Wt−1<br>) | zt<br>= f (xt<br>;Wt<br>)                  | O(1) |

Figure 3. Top: A generic sequence modeling layer expressed as a hidden state that transitions according to an update rule. All sequence modeling layers can be viewed as different instantiations of three components in this figure: the initial state, update rule and output rule. Bottom: Examples of sequence modeling layers and their instantiations of the three components. The naive TTT layer was shown in Figure [1.](#page-0-0) Self-attention has a hidden state growing with context, therefore growing cost per token. Both the naive RNN and TTT layer compress the growing context into a hidden state of fixed size, therefore their cost per token stays constant.

Wall-clock time. We apply two techniques to make TTT layers more efficient on modern GPUs and TPUs. First, similar to the standard practice of taking gradient steps on mini-batches of sequences during regular training for better parallelism, we use mini-batches of tokens during TTT. Second, we develop a dual form for operations inside each TTT mini-batch. The dual form is equivalent in output to the naive implementation, but trains more than 5× faster on our TPUs.

Contributions and limitations. The idea of using linear models as hidden states has already been well studied in DeltaNet [\[62,](#page-22-0) [83\]](#page-24-0). Since our first version was released, RNN layers with matrix (linear) hidden states have also been further advanced in Mamba 2 [\[17\]](#page-19-0) and Gated DeltaNet [\[82\]](#page-24-1). Compared to this line of work, our contribution is a practical framework that can instantiate arbitrary neural networks as hidden states. However, such instantiations can still require substantial wall-clock time, even after applying our improvements in efficiency. It remains to be seen whether our framework can produce instantiations that either overcome this limitation or offer benefits outweighing it.

## <span id="page-2-1"></span>2 Method

All sequence modeling layers can be viewed from the perspective of storing historic context into a hidden state, as shown in Figure [3.](#page-2-0) [1](#page-0-1) For example, RNN layers – such as LSTM [\[33\]](#page-20-1), RWKV [\[59\]](#page-22-1) and Mamba [\[27\]](#page-20-0) layers – compress context into a state of fixed size across time. This compression has two consequences. On one hand, mapping an input token *x<sup>t</sup>* to output token *z<sup>t</sup>* is efficient, because both the update rule and output rule take constant time per token. On the other hand, the performance of RNN layers in long context is limited by the expressive power of its hidden state *s<sup>t</sup>* .

Self-attention can also be viewed from the perspective above, except that its hidden state, commonly known as the Key-Value (KV) cache, is a list that grows linearly with *t*. Its update rule simply appends the current KV tuple to this list, and the output rule scans over all tuples up to *t* to form

<sup>1</sup> We define a sequence modeling layer as an autoregressive mapping from one sequence to another.

the attention matrix. The hidden state explicitly stores all historic context without compression, making self-attention more expressive than RNN layers for long context. However, scanning this linearly growing hidden state also takes linearly growing time per token.

To remain both efficient and expressive in long context, we need a better compression heuristic. Specifically, we need to compress thousands or potentially millions of tokens into a hidden state that can effectively capture their underlying structures and relationships. This might sound like a tall order, but all of us are actually already familiar with such a heuristic.

## <span id="page-3-1"></span>2.1 TTT as updating a hidden state

The process of parametric learning can be viewed as compressing a massive training set into the weights of a model. Specifically, we know that models trained with self-supervision can capture the underlying structures and relationships behind their training data [\[51\]](#page-22-2) – exactly what we need from a compression heuristic.

LLMs themselves are great examples. Trained with the self-supervised task of next-token prediction, their weights can be viewed as a compressed form of storage for existing knowledge on the internet. By querying LLMs, we can extract knowledge from their weights. More importantly, LLMs often exhibit a deep understanding of the semantic connections among existing knowledge to express new pieces of reasoning [\[1\]](#page-19-1).

Our key idea is to use self-supervised learning to compress the historic context *x*1*,..., x<sup>t</sup>* into a hidden state *s<sup>t</sup>* , by making the context an unlabeled dataset and the state a model. Concretely, the hidden state *s<sup>t</sup>* is now equivalent to *W<sup>t</sup>* , the weights of a model *f* , which can be a linear model, a small neural network, or anything else. The output rule is simply:

<span id="page-3-2"></span>
$$z_t = f(x_t; W_t). (1)$$

Intuitively, the output token is just the prediction on *x<sup>t</sup>* , made by *f* with the updated weights *W<sup>t</sup>* . The update rule is a step of gradient descent on some self-supervised loss *ℓ*:

$$W_t = W_{t-1} - \eta \,\nabla \ell(W_{t-1}; x_t), \tag{2}$$

with learning rate *η*. [2](#page-0-1) From the compression point of view, every heuristic needs to decide which input to remember or forget. Our *W* remembers inputs that produce large gradients – intuitively, inputs that make *W* learn a lot.

One choice of *ℓ* is reconstructing *x<sup>t</sup>* itself. To make the learning problem nontrivial, we first process *xt* into a corrupted input *x*˜*<sup>t</sup>* (details in Subsection [2.3\)](#page-4-0), then optimize:

<span id="page-3-0"></span>
$$\ell(W; x_t) = \|f(\tilde{x}_t; W) - x_t\|^2.$$
(3)

Similar to denoising autoencoders [\[77\]](#page-23-0), *f* needs to discover the correlations between dimensions of *xt* in order to reconstruct it from partial information *x*˜*<sup>t</sup>* . [3](#page-0-1) As shown in Figure [4,](#page-4-1) gradient descent is able to reduce *ℓ*, but cannot reduce it to zero. We discuss more sophisticated formulations of the self-supervised task in Subsection [2.3.](#page-4-0)

As with other RNN layers and self-attention, our algorithm that maps an input sequence *x*1*,..., x<sup>T</sup>* to output sequence *z*1*,..., z<sup>T</sup>* can be programmed into the forward pass of a sequence modeling layer, using the hidden state, update rule, and output rule above. Even at test time, our new layer still trains a different sequence of weights *W*1*,...,W<sup>T</sup>* for every input sequence. Therefore, we call it the *Test-Time Training (TTT) layer*.

<sup>2</sup> For now, consider *W*<sup>0</sup> = 0. We will discuss more sophisticated techniques for initializing *W* in Subsection [2.7.](#page-10-0)

<sup>3</sup> In past experiments, we have also tried adding another model *g* (decoder) after *f* (encoder), such that the reconstruction is produced by *g* ◦ *f* instead of only *f* itself. While this heftier design did slightly improve results, it made overall training less stable and added significant computational cost. Therefore we focus on the encoder-only design.

<span id="page-4-1"></span>![](_page_4_Figure_0.jpeg)

Figure 4. The self-supervised TTT loss  $\ell$  averaged over all test sequences of the form  $x_1, \dots, x_T$  where T = 2048, for the first three TTT layers in a network with 125M parameters. One step of gradient descent is able to reduce TTT loss from  $\ell(W_{t-1}; x_t)$  to  $\ell(W_t; x_t)$ . As t moves further along the test sequence,  $\ell(W_t; x_t)$  also improves further from  $\ell(W_0; x_t)$ . For visual clarity, loss values have been averaged over a sliding window of 10 timesteps. See Figure 14 (in Appendix) for complete results on all 12 layers.

### <span id="page-4-2"></span>2.2 Training a network with TTT layers

The forward pass of a TTT layer also has a corresponding backward pass. Our forward pass only consists of standard differentiable operators except the gradient operator  $\nabla$ . However,  $\nabla$  just maps one function to another, in this case  $\ell$  to  $\nabla \ell$ , and  $\nabla \ell$  is also composed of differentiable operators. Conceptually, calling backward on  $\nabla \ell$  means taking gradients of gradients – a well explored technique in meta-learning [54].

TTT layers have the same interface as RNN layers and self-attention, therefore can be replaced in any larger network architecture, which usually contains many of these sequence modeling layers. Training a network with TTT layers also works the same way as training any other language model, such as a Transformer. The same data, recipe, and objective such as next-token prediction can be used to optimize parameters of the rest of the network.

We refer to training the larger network as the *outer loop*, and training W within each TTT layer as the *inner loop*. An important difference between the two nested learning problems is that the inner-loop gradient  $\nabla \ell$  is taken w.r.t. W, the parameters of f, while the outer-loop gradient is taken w.r.t the parameters of the rest of the network, which we will denote by  $\theta_{\text{rest}}$ . Throughout this paper, outer-loop parameters are always denoted by  $\theta$  with various subscripts.

So far, the TTT layer has no outer-loop parameters, in contrast to other RNN layers and self-attention. In Subsection 2.3, we add outer-loop parameters to the TTT layer to improve its self-supervised task. Then in Subsection 2.4 and 2.5, we discuss two ways to improve the wall-clock time of TTT layers.

#### <span id="page-4-0"></span>2.3 Learning a self-supervised task for TTT

Arguably the most important part of TTT is the self-supervised task, because it determines the kind of features that W will learn from the test sequence. So how should we design this task? The final goal of TTT is for  $z_t = f(x_t; W_t)$  to perform well on language modeling. Instead of handcrafting a self-supervised task from human priors, we take a more end-to-end approach – directly optimizing the self-supervised task for the final goal of next-token prediction.

Concretely, we learn the self-supervised task as part of the outer loop. Starting from the naive reconstruction task in Equation 3, we add some outer-loop parameters to make this task learnable. In Subsection 2.1, we did not specify the corruption that produces  $\tilde{x}_t$  from  $x_t$ . One design is to make it a low-rank projection  $\tilde{x}_t = \theta_K x_t$ , where  $\theta_K$  is a learnable matrix. Following the terminology of multi-view reconstruction,  $\theta_K x_t$  is called a *training view* [13].

<sup>&</sup>lt;sup>4</sup> The subscript *K* hints at a connection to self-attention, as we will establish in Subsection 2.6.

```
class TTT_Layer(nn.Module):
                                              class Learner():
  def __init__(self):
                                                def __init__(self, task):
    self.task = Task()
                                                  self.task = task
                                                  # Linear here, but can be any model
                                                  self.model = Linear()
 def forward(self, in_seq):
    state = Learner(self.task)
                                                  # online GD here for simplicity
                                                  self.optim = OGD()
   out\_seq = []
    for tok in in_seq:
      state.train(tok)
                                                def train(self, x):
      out_seq.append(state.predict(tok))
                                                  # grad function wrt first arg
                                                  # of loss, which is self.model
    return out_seq
                                                  grad_fn = grad(self.task.loss)
                                                  # calculate inner-loop grad
class Task(nn.Module):
  def __init__(self):
                                                  grad_in = grad_fn(self.model, x)
    self.theta_K = nn.Param((d1, d2))
    self.theta_V = nn.Param((d1, d2))
                                                  # starting from current params,
    self.theta_Q = nn.Param((d1, d2))
                                                  # step in direction of grad_in,
                                                  self.optim.step(self.model, grad_in)
  def loss(self, f, x):
    train_view = self.theta_K @ x
                                                def predict(self, x):
    label_view = self.theta_V @ x
                                                  test_view = self.task.theta_Q @ x
    return MSE(f(train_view), label_view)
                                                  return self.model(test_view)
```

Figure 5. Naive implementation of a TTT layer with a linear model and online GD in the style of PyTorch. TTT\_Layer can be dropped into a larger network like other sequence modeling layers. Training the network will optimize the parameters of Task in TTT\_Layer, because both are subclasses of nn.Module. Since Learner is not a subclass of nn.Module, state.model is updated manually in the inner loop for each call of state.train. For simplicity, we sometimes overload model as model.parameters.

Moreover, perhaps not all the information in  $x_t$  is worth remembering, so the reconstruction label can be another low-rank projection  $\theta_V x_t$  instead of  $x_t$ . Here  $\theta_V x_t$  is called the *label view*, where  $\theta_V$  is also learnable. In summary, our new self-supervised loss is:

<span id="page-5-0"></span>
$$\ell(W; x_t) = \left\| f(\theta_K x_t; W) - \theta_V x_t \right\|^2. \tag{4}$$

Since both W and various  $\theta$ s appear together in Equation 4, we emphasize again their difference in nature. In the inner loop, only W is optimized, therefore written as an argument of  $\ell$ ; the  $\theta$ s are "hyper-parameters" of this loss function. In the outer loop,  $\theta_K, \theta_V, \theta_Q$  are optimized alongside  $\theta_{\text{rest}}$ , and W is merely a hidden state, not a parameter. Figure 5 illustrates this difference with code, where  $\theta_K$  and  $\theta_V$  are implemented as parameters of the TTT layer, analogous to the Key and Value parameters of self-attention.

Lastly, the training view  $\theta_K x_t$  has fewer dimensions than  $x_t$ , so we can no longer use the output rule in Equation 1. The simplest solution is to create a *test view*  $\theta_O x_t$ , and change our output rule to:

<span id="page-5-2"></span>
$$z_t = f\left(\theta_Q x_t; W_t\right). \tag{5}$$

This solution has an additional benefit. The training and label views specify the information in  $x_t$  that is compressed into  $W_t$  and propagated forward through time. The test view specifies potentially different information that is mapped to the current output token  $z_t$  and propagated forward through network layers, therefore adds more flexibility to the self-supervised task.

Altogether, the set of all possible choices for  $\theta_K$ ,  $\theta_Q$ ,  $\theta_V$  induces a family of multi-view reconstruction tasks, and the outer loop can be interpreted as selecting a task from this family. Here we have designed all views as linear projections for simplicity. Future work might experiment with more flexible transformations, or bigger and different families of self-supervised tasks.

<span id="page-6-2"></span>![](_page_6_Figure_0.jpeg)

Figure 6. High-level computation graph of the first TTT mini-batch, where nodes are variables and edges are computations. The blue nodes are input variables, and yellow are output. **Subsection 2.4**: Since  $G_1, \ldots, G_b$  are not connected, they have no sequential dependency on each other, therefore can be computed in parallel. **Subsection 2.5**: We do not actually materialize the white nodes – the intermediate  $G_1$  and  $G_2$  are not compute the output variables in the dual form.

#### <span id="page-6-0"></span>2.4 Parallelization with mini-batch TTT

The naive TTT layer developed so far is already efficient in the number of floating point operations (FLOPs). However, its update rule  $W_t = W_{t-1} - \eta \nabla l(W_{t-1}; x_t)$  cannot be parallelized, because  $W_t$  depends on  $W_{t-1}$  in two places: before the minus sign and inside  $\nabla l$ . Since  $\nabla l$  contains the bulk of the computation, we focus on making this second part parallel.

We approach this systems challenge through concepts in the TTT framework. There are many variants of gradient descent (GD). The general update rule of GD can be expressed as:

<span id="page-6-1"></span>
$$W_t = W_{t-1} - \eta G_t = W_0 - \eta \sum_{s=1}^t G_s,$$
 (6)

where  $G_t$  is the descent direction. Note that once we have calculated  $G_t$  for t = 1, ..., T, we can then obtain all the  $W_t$ s through a cumsum by the second half of Equation 6. Our naive update rule, known as online gradient descent, uses  $G_t = \nabla l(W_{t-1}; x_t)$ .

To parallelize  $G_t$  for t = 1, ..., T, we can take all of them w.r.t.  $W_0$ . This variant with  $G_t = \nabla \ell(W_0; x_t)$  is known as *batch gradient descent*, since  $\sum_{s=1}^t \nabla \ell(W_0; x_s)$  is the same as the gradient w.r.t.  $W_0$  over  $x_1, ..., x_t$  as a batch. However, in batch GD,  $W_t$  is effectively only one gradient step away from  $W_0$ , in contrast to online GD, where  $W_t$  is t steps away from  $W_0$ . Therefore, batch GD has a smaller effective search space, which ends up hurting performance for language modeling.

Our proposed solution – *mini-batch gradient descent* – is shown in Figure 6. Denote the TTT batch size by b. We use  $G_t = \nabla \ell(W_{t'}; x_t)$ , where t' = t - mod(t, b) is the last timestep of the previous mini-batch (or 0 for the first mini-batch), so we can parallelize b gradient computations at a time. Empirically, b controls a trade-off between speed and quality, as shown in Figure 7. We chose b = 16 for all experiments in this paper.

In summary, there are two potential channels to propagate information from  $W_s$  to  $W_t$  where s < t: cumsum and the gradient operator. The cumsum is always active, but the gradient channel is only active when  $W_s$  is from a previous mini-batch. Different variants of gradient descent only affect the gradient channel, *i.e.*, the descent direction  $G_t$ , specifically w.r.t. which  $W_t$  the gradient is taken. However, the descent step  $W_t = W_{t-1} - \eta G_t$  always starts from  $W_{t-1}$ , due to the autoregressive nature of the update rule, which is orthogonal to the choice of  $G_t$ .

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 7. Ablations on TTT mini-batch size b, where b=1 is online GD and b=T is batch GD. We choose b=16 for all experiments in this paper. **Left**: Smaller b improves perplexity since more GD steps are taken.<sup>5</sup> The perplexity of 11.09 at b=16 corresponds to the final result of TTT-Linear in Figure 10. **Right**: Forward time in dual form, with context length T=2048. Total time (orange) can be decomposed into time for computing the Ws at the end of every mini-batch (blue) and time for  $z_1, \ldots, z_T$  (orange – blue).<sup>6</sup> Time complexity for the Ws is  $O(T \times d^2)$ , constant in b, but the blue line decreases as larger b allows more parallelization until hardware utilization saturates. Time complexity for  $z_1, \ldots, z_T$  is  $O(T \times b \times d)$ , so the orange line first decreases with more parallelization, then increases as the extra computation for  $z_1, \ldots, z_T$  becomes dominant.

#### <span id="page-7-0"></span>2.5 Dual form

The parallelization introduced above is necessary but not sufficient for efficiency in wall-clock time. Modern accelerators specialize in matrix-matrix multiplications, known as matmuls. For example, the NVIDIA A100 GPU contains highly optimized units called TensorCores that can only perform a single operation – multiplying two matrices each of size  $16 \times 16$ . Without enough of these matmuls, the TensorCores are idle, and most of the potential for the A100 is unrealized.

Unfortunately, the TTT layer developed so far even with mini-batch still has very few matmuls. Consider the simplest case of  $\ell$ , where  $\theta_K = \theta_V = \theta_Q = I$ , for only the first TTT mini-batch of size b. In addition, consider f as a linear model. Copying Equation 3, our loss at time t is:

$$\ell(W_0; x_t) = \|f(x_t; W_0) - x_t\|^2 = \|W_0 x_t - x_t\|^2.$$

As discussed in Subsection 2.4, we can parallelize the computation of:

$$G_t = \nabla \ell (W_0; x_t) = 2(W_0 x_t - x_t) x_t^T,$$

for t = 1,...,b. However, we cannot compute all b of the  $G_t$ s through a single matmul. Instead, we need b outer products to compute them one by one. To make matters worse, for each  $x_t \in \mathbb{R}^d$ ,  $G_t$  is  $d \times d$ , which incurs much heavier memory footprint and I/O cost than  $x_t$  for large d.

To solve these two problems, we make a simple observation: We do not actually need to materialize  $G_1, ..., G_b$  as long as we can compute  $W_b$  at the end of the mini-batch, and the output tokens  $z_1, ..., z_b$  (see Figure 6). Now we demonstrate these computations with the simplified TTT-Linear case above. Denote  $X = [x_1, ..., x_b]$ , then:

$$W_b = W_0 - \eta \sum_{t=1}^b G_t = W_0 - 2\eta \sum_{t=1}^b (W_0 x_t - x_t) x_t^T = W_0 - 2\eta (W_0 X - X) X^T.$$

 $<sup>^{5}</sup>$  In theory, b can potentially be too small such that the variance between mini-batches is too high, hurting optimization. However, we have not observed such an effect in practice.

<sup>&</sup>lt;sup>6</sup> For Figure 7, we use a single TTT layer in TTT-Linear 1.3B, implemented in pure PyTorch. Our fused kernel significantly improves time efficiency, but makes it difficult to cleanly decompose the time for computing  $W_b$  vs.  $z_1, \ldots, z_b$ .

<span id="page-8-2"></span>![](_page_8_Figure_0.jpeg)

Figure 8. Parametric learners need to define two attributes: model and optimizer (left), and each learner uniquely induces a TTT layer (right). Two of the induced TTT layers: TTT-Linear and TTT-MLP, are proposed in this paper. The TTT layer with a linear model and batch GD is equivalent to linear attention [44].

So  $W_b$  can be conveniently computed with a matmul. To compute  $Z = [z_1, ..., z_b]$ , we know that:

<span id="page-8-1"></span>
$$z_t = f(x_t; W_t) = W_t x_t = \left(W_0 - \eta \sum_{s=1}^t G_t\right) x_t = W_0 x_t - 2\eta \sum_{s=1}^t (W_0 x_s - x_s) x_s^T x_t.$$
 (7)

Denote  $\delta_t = \sum_{s=1}^t (W_0 x_s - x_s) x_s^T x_t$  and the matrix  $\Delta = [\delta_1, \dots, \delta_b]$ . We can derive that:

$$\Delta = (W_0 X - X) \operatorname{mask}(X^T X), \tag{8}$$

where mask is the upper triangular mask with zeros (similar to the attention mask, but with zeros instead of infinities), and the term  $W_0X-X$  can be reused from the computation of  $W_b$ . Now  $\Delta$  is also conveniently computed with matmuls. Plugging  $\Delta$  back into Equation 7, we obtain  $Z=W_0X-2\eta\Delta$ .

We call this procedure the *dual form*, in contrast to the *primal form* before this subsection, where the Gs and Ws are explicitly materialized. As discussed, the two forms are equivalent in output. The terminology of primal and dual follows prior work that has explored similar mathematical formulations outside of TTT [35, 7, 61]. In Appendix A, we show that the dual form still works when f is a neural network with nonlinear layers, except with more complicated notation.

Time complexity of the primal form within a TTT mini-batch is  $O(b \times d^2)$ . Time complexity of the dual form is  $O(b \times d^2)$  for computing  $W_b$  alone, then an additional  $O(b^2 \times d)$  for computing  $z_1, \ldots, z_b$ . Compared to the primal, the dual form sacrifices theoretical complexity for hardware utilization. In practice, d is typically a few hundred and b is chosen to be only 16. As a consequence, wall-clock time for computing  $z_1, \ldots, z_b$  is relatively small, as observed in the right panel of Figure 7. In our JAX implementation, training with the dual form is more than  $5 \times$  faster than with primal.

#### <span id="page-8-0"></span>2.6 Theoretical equivalences

In Subsection 2.1, we mentioned that f can be a linear model or a neural network. In Subsection 2.4, we also discussed three variants of the update rule: online GD, batch GD, and mini-batch GD. Each of these  $2 \times 3$  combinations induces a different instantiation of the TTT layer, as illustrated in Figure 8. We now show that among these induced instantiations, the TTT layer with a linear model and batch GD is equivalent to linear attention [44], a widely known RNN layer.<sup>7</sup>

 $<sup>^7</sup>$  In a nutshell, linear attention [44] is simply self-attention without the softmax. Recall the definition of self-attention:  $z_t = V_t \text{softmax}\left(K_t^T q_t\right)$ . Without softmax, this becomes  $z_t = V_t\left(K_t^T q_t\right) = \sum_{s=1}^t v_s k_s^T q_t$ , which is the simplest formulation of linear attention. Similar to other RNN layers, it can be written in a recurrent form, where  $\sum_{s=1}^t v_s k_s^T$  is the hidden state. Since  $\sum_{s=1}^t v_s k_s^T$  can be computed in a cumsum for every  $t=1,\ldots,T$ , linear attention also has linear complexity w.r.t. T.

<span id="page-9-3"></span>![](_page_9_Figure_0.jpeg)

Figure 9. RNN layers and TTT layers are both subsets of sequence modeling layers. RNN layers have a hidden state that is fixed in size across time. TTT layers with parametric learners are also RNN layers, since their hidden state is also fixed in size. TTT layers with nonparametric learners can represent self-attention, as discussed in Subsection 2.6.

<span id="page-9-1"></span>**Theorem 1.** Consider the TTT layer with f(x) = Wx as the inner-loop model, batch gradient descent with  $\eta = 1/2$  as the update rule, and  $W_0 = 0$ . Then, given the same input sequence  $x_1, \ldots, x_T$ , the output rule defined in Equation 5 produces the same output sequence  $z_1, \ldots, z_T$  as linear attention.

*Proof.* By definition of  $\ell$  in Equation 4,  $\nabla \ell(W_0; x_t) = -2(\theta_V x_t)(\theta_K x_t)^T$ . By definition of batch GD in Equation 6:

$$W_{t} = W_{t-1} - \eta \nabla \ell(W_{0}; x_{t}) = W_{0} - \eta \sum_{s=1}^{t} \nabla \ell(W_{0}; x_{s}) = \sum_{s=1}^{t} (\theta_{V} x_{s}) (\theta_{K} x_{s})^{T}.$$

Plugging  $W_t$  into the output rule in Equation 5, we obtain the output token:

$$z_t = f\left(\theta_Q x_t; W_t\right) = \sum_{s=1}^t (\theta_V x_s) (\theta_K x_s)^T (\theta_Q x_t),$$

which is the definition of linear attention.

In Table 1, we first empirically verify the equivalence above with an improved implementation of linear attention.<sup>8</sup> Then, to illustrate the contribution of each of our components (including some that will be introduced in the next subsection), we add them row by row to the TTT layer that is equivalent to linear attention, and ultimately obtain our proposed instantiation called *TTT-Linear*. The change from batch GD to mini-batch GD contributes the most improvement by a large margin.

While the space of models  $\times$  optimizers in Figure 8 is already large, machine learning is much richer than optimizing the parameters  $W_t$  of a model f. There are also nonparametric learners, such as nearest neighbors, support vector machines (SVMs), and kernel ridge regression. By definition, nonparametric learners do not have parameters  $W_t$ , and instead directly uses training data  $x_1, \ldots, x_t$ . Hence we use the notation  $f(x; x_1, \ldots, x_t)$ . We now show that for a particular nonparametric learner, the induced TTT layer is equivalent to self-attention.

<span id="page-9-2"></span>**Theorem 2.** Consider the TTT layer with the Nadaraya-Watson estimator [6, 11], defined as:

<span id="page-9-0"></span>
$$f(x; x_1, \dots, x_t) = \frac{1}{\sum_{s=1}^t \kappa(x, x_s)} \sum_{s=1}^t \kappa(x, x_s) \ y_s, \tag{9}$$

where  $y_s = \theta_V x_s$  is the label view discussed in Subsection 2.3, and

<span id="page-9-4"></span>
$$\kappa(x, x'; \theta_K, \theta_Q) \propto e^{(\theta_K x)^T \theta_Q x'}$$
 (10)

is a kernel with bandwidth hyper-parameters  $\theta_K$  and  $\theta_Q$ . Then given the same input sequence  $x_1,...,x_T$ , the output rule defined in Equation 5 produces the same output sequence  $z_1,...,z_T$  as self-attention.

 $<sup>^8</sup>$  The original formulation of linear attention in [44] contains a normalizer and a feature expansion on  $x_t$ , which can still be included in an equivalent TTT layer. However, prior work has found that these two additions can hurt performance [60], which we have verified in our own experiment (first vs. second row of Table 1). Therefore, we only construct a TTT layer equivalent to the simplest formulation of linear attention without the two additions.

<span id="page-10-1"></span>

| Configuration          | Ppl.  | Diff. |  |
|------------------------|-------|-------|--|
| Linear attention [44]  | 15.91 | -     |  |
| Linear attn. improved  | 15.23 | −0.68 |  |
| TTT equivalence        | 15.23 | 0     |  |
| + learnable W0         | 15.27 | +0.04 |  |
| + LN and residual in f | 14.05 | −1.22 |  |
| + mini-batch TTT       | 12.35 | −1.70 |  |
| + learnable η          | 11.99 | −0.36 |  |
| + Mamba backbone       | 11.09 | −0.90 |  |

Table 1. Ablations on improving from linear attention. All models here have 125M parameters, and are trained according to the recipe in Subsection [3.1.](#page-12-1) The last row, with perplexity 11.09, is the final result of TTT-Linear in Figure [10.](#page-12-0) Starting from the equivalence discussed in Subsection [2.6,](#page-8-0) learnable *W*0 hurts slightly, but the rows below cannot train stably without it. The biggest improvement comes from mini-batch TTT (changing from *b* = *T* = 2048 to *b* = 16). The second comes from instantiating the inner model *f* with LN and residual connection. Both of these designs would be difficult to come across without the conceptual framework of TTT.

*Proof.* Plugging *y<sup>s</sup>* and *κ* above into Equation [9](#page-9-0) gives us the definition of self-attention.

Appendix [B](#page-27-0) contains a detailed explanation of the Nadaraya-Watson estimator and kernel *κ* above. In contrast to Theorem [1,](#page-9-1) Theorem [2](#page-9-2) does not produce a different implementation from attention.

For the TTT layer above, the hidden state is *x*1*,..., x<sup>t</sup>* or a similar list of processed training data, the update rule adds *x<sup>t</sup>* to the list, and the output rule scans the list with *κ*. In previous subsections, our hidden state has been defined as *W<sup>t</sup>* , the update rule a gradient step, and the output rule a call to *f* . To unify these two constructions, we define a new abstraction called a learner, which uniquely induces a TTT layer.

Similar to its definition in standard machine learning packages [\[57\]](#page-22-6), all learners need to implement two methods: train and predict. Now we redefine the hidden state of the induced TTT layer as the internal storage of the learner, and the update and output rules as the train and predict methods. Under this new definition of TTT layers, both parametric learners such as that in Theorem [1](#page-9-1) and nonparametric learners such as that in Theorem [2](#page-9-2) can be included. Figure [9](#page-9-3) summarizes this general definition of TTT layers in the broader scope of all sequence modeling layers.

This general definition has an additional benefit for parametric learners: There can be more objects other than *W* in the internal storage of parametric learners, such as the optimizer state, which will also be included in the hidden state of the induced TTT layer. This extension allows TTT layers to use more sophisticated optimizers such as Adam [\[45\]](#page-21-3) in future work.

### <span id="page-10-0"></span>2.7 Implementation details

Instantiations of *f* . We propose two variants of TTT layers – TTT-Linear and TTT-MLP, differing only in their instantiations of *f* . For TTT-Linear, *f*lin(*x*) = *W x*, where *W* is square. For TTT-MLP, *f*MLP has two layers similar to the MLPs in Transformers. Specifically, the hidden dimension is 4× the input dimension, followed by a GELU activation [\[31\]](#page-20-2). For better stability during TTT, *f* always contains a Layer Normalization (LN) and residual connection. That is, *f* (*x*) = *x* + LN(*f*res(*x*))*,* where *f*res can be *f*lin or *f*MLP.

Learnable *W*0. The TTT initialization *W*<sup>0</sup> is shared between all sequences, even though subsequent weights *W*1*,...,W<sup>T</sup>* are different for each input sequence. Instead of setting *W*<sup>0</sup> = 0, we can learn it as part of the outer loop. Since outer-loop parameters are always denoted by *θ*s instead of *W* s, we assign an alias *θ*init = *W*0. In practice, *θ*init adds a negligible amount of parameters comparing to the reconstruction views *θK,θQ,θ<sup>V</sup>* , because both its input and output are low dimensional. Empirically, we observe that learning *W*<sup>0</sup> significantly improves training stability.

Learnable *η*. The learning rate is usually the most important hyper-parameter for gradient descent, so we experiment with learning the inner-loop learning rate *η* in Equation [6](#page-6-1) as part of the outer loop. We make *η* a function of the input token (therefore different across time) for additional flexibility. Concretely, we design *η*(*x*) = *η*base *σ*(*θ*lr ·*x*), where the learnable vector *θ*lr is an outer-loop parameter, *σ* is the sigmoid function, and the scalar *η*base is the base learning rate, set to 1 for TTT-Linear and 0.1 for TTT-MLP. Alternatively, *η*(*x*) can also be interpreted as a gate for ∇*ℓ*.

Backbone architecture. The cleanest way to integrate any RNN layer into a larger architecture would be to directly replace self-attention in a Transformer, known in this context as a backbone. However, existing RNNs such as Mamba [\[27\]](#page-20-0) and Griffin [\[18\]](#page-20-3) all use a different backbone from Transformers. Most notably, their backbone contains temporal convolutions before the RNN layers, which might help collect local information across time. After experimenting with the Mamba backbone, we find that it also improves perplexity for TTT layers, so we incorporate it into our proposed method. See Figure [13](#page-28-0) (in Appendix) for details.

## <span id="page-11-0"></span>3 Experiments

We evaluate TTT-Linear and TTT-MLP by comparing with two baselines – Transformer and Mamba, a modern RNN. Our main codebase is based on EasyLM [\[25\]](#page-20-4), an open-source project for training and serving LLMs in JAX. All experiments can be reproduced using the publicly available code and datasets provided at the bottom of the first page.

Datasets. Following the Mamba paper [\[27\]](#page-20-0), we perform standard experiments with 2k and 8k context lengths on the Pile [\[24\]](#page-20-5), a popular dataset of documents for training open-source LLMs [\[8\]](#page-19-6). However, the Pile contains few sequences of length greater than 8k [\[19\]](#page-20-6). To evaluate capabilities in long context, we also experiment with context lengths ranging from 1k to 32k in 2× increments, on a subset of the Pile called Books3, which has been widely used to train LLMs in long context [\[52,](#page-22-7) [3\]](#page-19-7).

Backbone architecture. As discussed in Subsection [2.7,](#page-10-0) Transformer and Mamba use different backbones, and TTT-Linear and TTT-MLP always use the Mamba backbone unless noted otherwise. As an ablation study, Figure [10](#page-12-0) and Figure [11](#page-13-0) contain TTT layers within the Transformer backbone. When a figure contains both the Transformer backbone and Mamba backbone, we denote them by *(T)* and *(M)*, respectively.

<span id="page-11-1"></span>Protocols. To ensure fairness to our baselines, we strictly follow the evaluation protocols in the Mamba paper when possible:

- For each evaluation setting (e.g., dataset, context length, and method), we experiment with four model sizes: 125M, 350M, 760M, and 1.3B parameters. For Mamba, the corresponding sizes are 130M, 370M, 790M, and 1.4B, as Mamba does not follow the Transformer configurations.
- All models are trained with the Chinchilla recipe[9](#page-0-1) described in the Mamba paper and reproduced in our Appendix [C.](#page-28-1) Our Transformer baseline, based on the Llama architecture [\[75\]](#page-23-1), also follows the baseline in the Mamba paper. As verification, our baselines can reproduce the numbers reported in the Mamba paper in their evaluation settings.[10](#page-0-1)

<sup>9</sup> The Chinchilla paper is another highly influential study of empirical scaling laws [\[34\]](#page-20-7). From large-scale experiments with many hyper-parameters, they observe that the compute-optimal models follow a particular training recipe. We only follow the Chinchilla recipe used in the Mamba paper, which may be slightly different from the original recipe in [\[34\]](#page-20-7).

<sup>10</sup> The only difference between our protocol and that in the Mamba paper is the tokenizer. The Mamba paper uses two different tokenizers – GPT-2 and GPT-NeoX – for various experiments. For consistency, we adhere to a single tokenizer throughout this paper and choose the Llama tokenizer [\[75\]](#page-23-1), which is the modern state-of-the-art.

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 10. Evaluations for context lengths 2k and 8k on the Pile. Details in Subsection 3.1. TTT-Linear has comparable performance as Mamba at 2k context, and better performance at 8k.

• We do not experiment with hybrid architectures (e.g. Griffin [18]), because our baselines are not hybrid. While hybrid architectures that use both self-attention and TTT layers may improve performance, they would reduce the clarity of our academic evaluation.

### <span id="page-12-1"></span>3.1 Short context: the Pile

From Figure 10, we make a few observations:

- At 2k context, TTT-Linear (M), Mamba, and Transformer have comparable performance, as the lines mostly overlap. TTT-MLP (M) performs slightly worse under large FLOP budgets. Even though TTT-MLP has better perplexity than TTT-Linear at every model size, the extra cost in FLOPs offsets the advantage.
- At 8k context, both TTT-Linear (M) and TTT-MLP (M) perform significantly better than Mamba, in contrast to the observation at 2k. Even TTT-MLP (T) with the Transformer backbone performs slightly better than Mamba around 1.3B. A robust phenomenon we observe throughout this paper is that as context length grows longer, the advantage of TTT layers over Mamba widens.
- At 8k context, Transformer still has good (if not the best) perplexity at every model size, but its line is not competitive because of the cost in FLOPs.

**Effect of backbone.** Switching the TTT layers from Mamba backbone into Transformer backbone has two effects. First, TTT layers with Mamba backbone perform better in our evaluations so far. Second, with Mamba backbone, TTT-MLP at best is only comparable to TTT-Linear; but with Transformer backbone, TTT-MLP is clearly better. We hypothesize that the temporal convolutions in the Mamba backbone help more when the sequence modeling layer has a less expressive hidden state. The linear model is less expressive than the MLP, therefore benefits more from the convolutions. We will revisit this hypothesis in the next subsection.

**Lack of linear fit.** The Chinchilla paper empirically observed that the compute-optimal models following their recipe fall onto a line in the log-log plot of FLOPs vs. perplexity, as is often the case for scaling law experiments [34]. However, we do not observe a clean linear fit in Figure 10 or Figure 11 (the analogous experiments in Books), not even for Transformers. This is not surprising

<span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

Figure 11. Evaluations for context lengths 2k and 32k on Books. Details in Subsection 3.2. Our complete results for context lengths 1k, 2k, 4k, 8k, 16k, 32k, including Transformer finetuning, are in Figure 15 (in Appendix). Most observations from the Pile still hold.

given the differences in dataset, context length, tokenizer, and architecture. Following the Mamba paper, we connect the points instead of fitting them with linear regression due to the large error. <sup>11</sup>

#### <span id="page-13-1"></span>3.2 Long context: Books

To evaluate capabilities in long context, we experiment with context lengths ranging from 1k to 32k in  $2\times$  increments, using a popular subset of the Pile called Books3. The training recipe here is the same as that for Pile.<sup>12</sup> From the subset of results in Figure 11, we make a few observations:

- At 2k context on Books, all the observations from Pile 2k still hold, except that Mamba now performs slightly better than TTT-Linear (whereas their lines roughly overlapped for Pile 2k).
- At 32k context, both TTT-Linear (M) and TTT-MLP (M) perform better than Mamba, similar to the observation from Pile 8k. Even TTT-MLP (T) with the Transformer backbone performs slightly better than Mamba at 32k context.
- TTT-MLP (T) is only slightly worse than TTT-MLP (M) at 1.3B scale. As discussed, it is hard to derive an empirical scaling law due to the lack of a clean linear fit. However, the strong trend for TTT-MLP (T) suggests that the Transformer backbone might be more suitable for larger models and longer context beyond our evaluations.

We only ablate the backbones for 2k and 32k due to the cost of training LLMs. For future work, we believe that given TTT layers with even more expressive hidden states, the Mamba backbone with temporal convolutions will become unnecessary.

**Transformer finetuning.** While we have been training Transformers from scratch following the Mamba paper, in practice this approach is rarely used for long context. The standard practice is to train a Transformer in short context, then finetune in long context. To reflect this practice, we add

<sup>&</sup>lt;sup>11</sup> Ideally, we would have rerun all the hyper-parameters and derived a potentially new recipe for each method based on our evaluation setting, following the process in the Chinchilla paper. If the new compute-optimal models do fall onto a line, we could then predict performance beyond the current FLOPs regime [43, 34]. However, this empirical study would require orders of magnitude more resources than ours.

 $<sup>^{12}</sup>$  Following the Mamba paper, we always use 0.5M tokens per training batch regardless of the context length. In other words, for context length T we have 0.5M/T sequences per batch (assume divisible).

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 12. Latency on an NVIDIA A100 GPU with 80G HBM and PCIe connections.

another baseline, *TF finetune*, for context lengths 4k and above. This baseline starts from the model trained (according to the Chinchilla recipe) on Books 2k, then uses 20% more tokens to finetune at the designated context length, following the Llama Long paper [81]. See details of the TF finetune recipe in Appendix C.

**Experiments in Figure 2 (right).** Compared to TTT-Linear, TTT-MLP with matched FLOPs performs worse at short context but better at long context. This observation matches our expectation that the MLP as hidden state is more expressive than the linear model: The larger capacity of a more expressive hidden state is well-utilized in long context (therefore an advantage), but redundant in short context (therefore a disadvantage in our setting with matched FLOPs). The Transformer in this figure is TF finetune, which is the stronger baseline in 32k context. Details of the experiments in Figure 2 are included in Appendix C.

Our complete results for context lengths 1k, 2k, 4k, 8k, 16k, 32k, including TF finetune, are in Figure 15 (in Appendix).

#### <span id="page-14-1"></span>3.3 Wall-clock time

LLM training and inference can be decomposed into forward, backward, and generate. Prompt processing during inference, also known as prefill, is the same operation as forward during training, except that the intermediate activations do not need to be stored for backward. Since both forward (during training and inference) and backward can be parallelized, we use the dual form. Generating new tokens, also known as decode, is inherently sequential, so we use the primal form.

Due to resource constraints, our experiments are written in JAX and run on TPUs. On a v5e-256 TPU pod, the Transformer baseline takes 0.30s per iteration of training at context 2k, while TTT-Linear takes 0.27s per iteration, already 10% faster without any systems optimization. However, Mamba (implemented in PyTorch, Triton, and CUDA) can only run on GPUs, so for fair comparison, we also rewrite our method into GPU kernels. We only write inference kernels for this work because the training kernel would require substantial effort and cannot be used on our TPUs.

Figure 12 shows the latency of our inference kernel for forward (prefill) and generate (decode). All models are 1.3B (1.4B for Mamba). As expected, time per token grows linearly for Transformer as the context length increases, but stays roughly constant for the other methods.<sup>13</sup> Note that our

 $<sup>^{13}</sup>$  We observe that the forward latency of the network increases slightly for TTT-Linear, TTT-MLP, and Mamba, even though the latency of each sequence modeling layer alone stays constant. Consider the operation  $\theta X$ , where  $\theta$  is  $d \times d$  and X is  $d \times T$ . Its latency (normalized over T) is expected to be constant, but in practice grows slightly with T. One possible cause of this phenomenon is the GPU throttling after T gets very large [30].

Transformer baseline is significantly faster that in the Mamba paper, because we use vLLM [\[49\]](#page-21-4), a state-of-the-art serving system, instead of the HuggingFace Transformer [\[80\]](#page-23-2).

## 4 Related Work

### 4.1 Learning at Test Time

The idea of learning at test time has a long history in machine learning. One of the earliest versions of this idea is called local learning (Bottou and Vapnik [\[9\]](#page-19-8)): For each test input, train on its neighbors before making a prediction. This procedure has been effectively applied to models ranging from SVMs [\[85\]](#page-24-3) to modern LLMs [\[29\]](#page-20-9).

Another early version of learning at test time is called *transductive learning* [\[22\]](#page-20-10). The principle of transduction, as stated by Vladimir Vapnik [\[76\]](#page-23-3), is to "... get the answer that you really need, but not a more general one." Practical implementations of transductive learning use test data to add constraints to the margin of SVMs [\[42,](#page-21-5) [16\]](#page-19-9). However, transductive learning usually needs multiple test instances to be empirically effective, unlike many instantiations of test-time training, which only need a test single instance (image, video, or natural language sequence) at a time.

In computer vision, the idea of learning at test time has been applied for decades to applications such as face detection [\[41\]](#page-21-6), object detection [\[56\]](#page-22-8), image super-resolution [\[68\]](#page-23-4), and 3D reconstruction [\[53\]](#page-22-9). More recently, the same idea has also been applied to natural language processing, where it is called dynamic evaluation [\[47,](#page-21-7) [48\]](#page-21-8). The basic approach is to directly finetune a language model on the test sequence, which often comes in the form of a prompt.

Next, we discuss two relevant lines of work in detail: test-time training and fast weights.

#### 4.1.1 Test-Time Training

The core idea of *Test-Time Training* (TTT) is that each test instance defines its own learning problem, where this test instance alone is the target of generalization [\[72\]](#page-23-5). Concretely, for each test instance *x*, the conventional practice is to predict *f* (*x*), using a predictor *f* that is optimized for all training instances on average. TTT first formulates a learning problem defined by *x*, then trains a model *f<sup>x</sup>* on *x* (often with *f* as initialization), and predicts *f<sup>x</sup>* (*x*).

Since the test instance comes without its label, the learning problem can only be formulated with a self-supervised task. Prior work has shown that TTT with reconstruction significantly improves performance especially on outliers [\[23\]](#page-20-11). Improvements become even more pronounced when testing on video frames that arrive in a stream and TTT is autoregressive [\[79\]](#page-23-6), as *f<sup>t</sup>* is trained on past frames *x*1*,..., x<sup>t</sup>* . The autoregressive connection makes [\[79\]](#page-23-6) most relevant to our paper.

Conceptually, the biggest difference between our paper and prior work is that our reconstruction task is learned in an outer loop, instead of handcrafted with human priors. Follow-up work has explored applications such as robot manipulation [\[28\]](#page-20-12) and locomotion [\[71\]](#page-23-7), among others, that often require different designs for the self-supervised task. In a preliminary manuscript [\[70\]](#page-23-8), we explore the idea of learning to (learn at test time) from the perspective of TTT with reconstruction, with experiments in object recognition.

#### 4.1.2 Fast Weights and Fast Weight Programmers

The general idea of *fast weights* is to update the parameters of a "fast" model on only the most relevant data, as opposed to the conventional practice of updating a "slow" model on all data [\[74\]](#page-23-9). This idea has existed since the 1980s [\[20,](#page-20-13) [32,](#page-20-14) [78\]](#page-23-10). The most relevant data often includes the test instance itself, therefore TTT can be viewed as a special case of fast weights. Compared to fast

weights, TTT embraces the idea of formulating an explicit learning problem, where the test instance is the target of generalization. Our update rule is also an explicit step of optimization.

The idea of *fast weight programmers* (FWPs) is to update the fast weights at test time with a "slow" model that is updated less frequently, if at all [\[65\]](#page-22-10). Our inner-loop weights *W* can be viewed as "fast" and outer-loop weights *θ* as "slow". Therefore, networks containing TTT layers can be viewed as a special case of FWPs [\[46\]](#page-21-9), similar to how TTT can be viewed as a special case of fast weights. Notably, one instantiation in Irie et al. [\[38\]](#page-21-10) makes the fast weights an MLP, preceding our TTT-MLP.

Many other modern RNN layers such as linear attention [\[44,](#page-21-1) [63\]](#page-22-11) and DeltaNet [\[62,](#page-22-0) [83\]](#page-24-0) are also inspired by the idea of FWPs. Given their relevance to our work, we discuss these modern RNN layers in detail in the next subsection. For the rest of this subsection, we briefly outline a few other forms of FWPs. Clark et al. [\[15\]](#page-19-10) give a Transformer a final layer of fast weights, whose initialization is trained as slow weights. Irie et al. [\[39\]](#page-21-11) design the fast weights to be programmed by themselves, which can be interpreted as recursive self-improvement. In addition, [\[40\]](#page-21-12) builds an image generator using the images as fast weights, [\[37\]](#page-21-13) applies continuous-time extensions of FWPs to time-series classification, while [\[36\]](#page-21-14) and [\[26\]](#page-20-15) demonstrate how the choice of update rules affects the expressiveness of FWPs on formal language recognition tasks.

## 4.2 Modern RNN layers

Our baseline, Mamba [\[27\]](#page-20-0), is only one of the many recent RNN layers that inherit the linear (matrix) hidden states of linear attention [\[44,](#page-21-1) [63\]](#page-22-11). Some more recent examples are RWKV [\[58,](#page-22-12) [59\]](#page-22-1), xLSTM [\[4\]](#page-19-11), and Gated Linear Attention (GLA) [\[82\]](#page-24-1). The most relevant work is DeltaNet [\[62\]](#page-22-0), which is equivalent to TTT-Linear with inner-loop mini-batch size 1, without the Layer Norm and residual connection. In [\[83\]](#page-24-0), Yang et al. further improve the performance of DeltaNet and enable parallelized updates across tokens (in our terms, across inner loop mini-batches). Since our first version was released, RNN layers with matrix (linear) hidden states have also been further advanced in Mamba 2 [\[17\]](#page-19-0) and Gated DeltaNet [\[82\]](#page-24-1).

Compared to this line of work, our contribution is a practical framework that can instantiate arbitrary neural networks as hidden states. However, such instantiations can still require substantial wall-clock time, even after applying our improvements in efficiency. For example, TTT-MLP is effective in terms of FLOPs, as shown in Figure [2.](#page-1-0) But the additional complexity of the MLP structure increases wall-clock time much more relative to FLOPs, as shown in Figure [12.](#page-14-0) It remains to be seen whether our framework can produce instantiations that either overcome this limitation or offer benefits outweighing it.

### 4.3 Learning to Learn

For decades, researchers have been arguing that learning to learn, also known as meta-learning or bi-level optimization, should be a critical component of intelligence [\[64,](#page-22-13) [5,](#page-19-12) [73,](#page-23-11) [50\]](#page-21-15). In prior work such as [\[2\]](#page-19-13), [\[21\]](#page-20-16) and [\[55\]](#page-22-14), the inner loop learns from an entire dataset at a time instead of a sequence, so the outer loop needs a collection of datasets or tasks. In short, the outer loop is "one level above" regular training. Since it is hard to collect millions of datasets, this outer loop is hard to scale.

In contrast, for TTT, each sequence itself is a dataset and defines its own generalization problem. The inner loop is "one level below" regular training, so our outer loop is only another solution to the canonical problem of supervised learning, instead of a new problem setting like generalization across datasets. As illustrated in Table [2,](#page-17-0) our outer loop is "at the same level" as regular training. This makes our outer loop easier to scale.

<span id="page-17-0"></span>

|               | Inner loop              | Outer loop                         | Subsection |  |
|---------------|-------------------------|------------------------------------|------------|--|
| Piece of data | Token xt                | Sequence x1,, xT                   |            |  |
| Training set  | Sequence x1,, xT        | Dataset of sequences, e.g., Books  | 2.1, 2.2   |  |
| Objective     | Reconstruction (loss ℓ) | Next-token prediction              |            |  |
| Parameters    | W (weights of f )       | (rest of the network)<br>θrest     |            |  |
|               |                         | (reconstruction views)<br>θK,θQ,θV | 2.3        |  |
|               |                         | and θlr<br>θinit                   | 2.7        |  |

Table 2. In summary, our paper reformulates supervised learning as learning to learn, with two nested loops. Highlighted rows of the outer loop are the same as in the regular training. Parameters of the outer loop become hyper-parameters of the inner loop. Intuitively, the inner loop, *i.e.* TTT, is "one level below" regular training.

## 5 Discussion

We have reformulated the canonical problem of supervised learning as learning to (learn at test time). Our formulation produces an alternative conceptual framework for building what is traditionally known as network architectures. We summarize our current instantiation in Table [2.](#page-17-0)

Future work. The search space for effective instantiations inside this framework is huge, and our paper has only taken a baby step. Fortunately, if our perspective holds, then heuristics from regular training can transfer to test-time training, and search can be efficient. Next we outline some especially promising directions for future work:

- Outer-loop parameterization. There are many other ways to parameterize a family of multi-view reconstruction tasks, or perhaps a more general family of self-supervised tasks. It would be a big coincidence if the first one we have tried turns out to be the best.
- Systems optimization. Our systems optimization in Subsection [3.3](#page-14-1) has been preliminary at best, and there are many ways to improve it. In addition, pipeline parallelism through time might allow us to process long sequences of millions of tokens on multiple devices together.
- Longer context and larger models. Constrained by our academic resources, we have not trained with millions or billions in context length, which would also require larger models according to Figure [16.](#page-32-0) The advantage of TTT layers should become more pronounced in longer context.
- More ambitious instantiations of *f* . When context length becomes longer, *f* would also need to be larger. For video tasks and embodied agents, whose context length can easily scale up to millions or billions, *f* could be a convolutional neural network.
- Multi-level learning to learn. If *f* itself is a self-attention layer, then by Theorem [2](#page-9-2) it can be interpreted as yet another inner loop nested inside the existing one. In this fashion, we can potentially build many levels of nested learning problems. Versions of this idea have already been explored in [\[38\]](#page-21-10) and [\[39\]](#page-21-11).

Why do we study TTT? First a more basic question: Why study AI? For some of us, AI is a playground to probe about the nature of human intelligence. Prior work often tries to model human learning with machine learning, where training is on a shuffled dataset with i.i.d. instances, and inference is on a separate test set. However, humans do not naturally learn with i.i.d. instances or have a train-test split. We believe that human learning has a more promising connection with TTT, our inner loop, whose data is a potentially very long sequence with strong temporal dependencies, and any piece of data can be used for both training and testing. This is why we study TTT.

## Author Contributions

Yu Sun started this project with Xinhao Li in November 2022, and has been working on it full-time since June 2023. Yu proposed the conceptual framework of the project, designed mini-batch TTT and the dual form, wrote the paper with help from others, and led the daily operations of the team.

Xinhao Li started this project with Yu Sun in November 2022, and has been working on it full-time since then. Xinhao and Karan co-led the development of our current codebase. Before March 2024, Xinhao was the primary contributor to our earlier codebases that shaped this project. Xinhao made significant contributions to the project direction in discussions.

Karan Dalal joined this project full-time in June 2023. In collaboration with Xinhao, he co-led the development of our current codebase. Karan managed the experiments in Section [3,](#page-11-0) helped write the paper, and made significant contributions to the project direction in discussions.

Jiarui Xu joined this project in March 2024. He led our architectural development since he joined, and made significant contributions to the project direction in discussions.

Arjun Vikram joined this project in September 2023. He made significant contributions to our systems optimization, as well as current and earlier codebases.

Genghan Zhang joined this project in January 2024. He provided critical insights and made significant improvements to our systems optimization.

Yann Dubois joined this project in February 2024. He proposed our current instantiation of *f* , and made significant contributions to the project direction in discussions.

Xinlei Chen and Xiaolong Wang have been supporting this project since November 2022, and the direction of test-time training for many years. Without their support in compute and organization, this project could not have survived its early stage. They gave invaluable advice to our experiments.

Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin have been supporting this project since May 2023. They gave invaluable advice to our experiments and presentation. For example, Sanmi suggested us to focus on TTT-Linear, Tatsu suggested the experiments in Figure [2](#page-1-0) (left), and Carlos outlined Section [2.](#page-2-1)

# Acknowledgements

Part of the compute for this project is generously supported by the Google TPU Research Cloud program and Hyperbolic Labs. XW is supported, in part, by the Amazon Research Award, the Cisco Faculty Award and the Qualcomm Innovation Fellowship. SK acknowledges support by NSF 2046795 and 2205329, NIFA award 2020-67021-32799, the Alfred P. Sloan Foundation, and Google Inc. TH is supported by a Sony Faculty Innovation Award and a gift from Panasonic. CG acknowledges support by the Air Force Office of Scientific Research (AFOSR), FA9550-20-1-0427, Stanford Human-Centered Artificial Intelligence (HAI) Institute, and gifts from Google and IBM.

We would like to thank Rohan Taori, Xuechen Li, Allan Zhou, Ke Chen, and Guandao Yang for many helpful discussions, Menghao Guo for help with code release, Xinyang Geng for help with EasyLM, Hao Liu for help with the LWM codebase, David Hall for help with Levanter, Yossi Gandelsman and Yutong Bai for help at an early stage of the project, Mert Yuksekgonul for help with figures in the paper, Horace He, Ben Spector, and Azalia Mirhoseini for help with systems, Sharad Vikram and Roy Frostig for answering our questions about JAX and Pallas, Albert Gu and Tri Dao for helping us reproduce experiments in the Mamba paper, and Kilian Weinberger and Percy Liang for advice on presentation. Yu Sun is grateful to his PhD advisors, Alexei A. Efros and Moritz Hardt, for their many insights from years ago that eventually became part of this paper.

## References

- <span id="page-19-1"></span>[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.
- <span id="page-19-13"></span>[2] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. *Advances in neural information processing systems*, 29, 2016.
- <span id="page-19-7"></span>[3] Authors Guild. You just found out your book was used to train ai. now what?, 2023. Accessed: 2024-06-24.
- <span id="page-19-11"></span>[4] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended long short-term memory. *arXiv preprint arXiv:2405.04517*, 2024.
- <span id="page-19-12"></span>[5] Yoshua Bengio, Samy Bengio, and Jocelyn Cloutier. *Learning a synaptic learning rule*. Citeseer, 1990.
- <span id="page-19-4"></span>[6] Hermanus Josephus Bierens. The nadaraya-watson kernel regression function estimator. *(Serie Research Memoranda; No. 1988-58). Faculty of Economics and Business Administration, Vrije Universiteit Amsterdam.*, 1988.
- <span id="page-19-3"></span>[7] Christopher M Bishop and Nasser M Nasrabadi. *Pattern recognition and machine learning*, volume 4. Springer, 2006.
- <span id="page-19-6"></span>[8] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. *arXiv preprint arXiv:2204.06745*, 2022.
- <span id="page-19-8"></span>[9] Léon Bottou and Vladimir Vapnik. Local learning algorithms. *Neural computation*, 4(6):888–900, 1992.
- <span id="page-19-15"></span>[10] Leo Breiman, William Meisel, and Edward Purcell. Variable kernel estimates of multivariate densities. *Technometrics*, 19(2):135–144, 1977.
- <span id="page-19-5"></span>[11] Zongwu Cai. Weighted nadaraya–watson regression estimation. *Statistics & probability letters*, 51(3):307–318, 2001.
- <span id="page-19-16"></span>[12] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost, 2016.
- <span id="page-19-2"></span>[13] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. *arXiv preprint arXiv:2003.04297*, 2020.
- <span id="page-19-14"></span>[14] Yen-Chi Chen. A tutorial on kernel density estimation and recent advances. *Biostatistics & Epidemiology*, 1(1):161–187, 2017.
- <span id="page-19-10"></span>[15] Kevin Clark, Kelvin Guu, Ming-Wei Chang, Panupong Pasupat, Geoffrey Hinton, and Mohammad Norouzi. Meta-learning fast weight language models. *arXiv preprint arXiv:2212.02475*, 2022.
- <span id="page-19-9"></span>[16] Ronan Collobert, Fabian Sinz, Jason Weston, Léon Bottou, and Thorsten Joachims. Large scale transductive svms. *Journal of Machine Learning Research*, 7(8), 2006.
- <span id="page-19-0"></span>[17] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. *arXiv preprint arXiv:2405.21060*, 2024.

- <span id="page-20-3"></span>[18] Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, et al. Griffin: Mixing gated linear recurrences with local attention for efficient language models. *arXiv preprint arXiv:2402.19427*, 2024.
- <span id="page-20-6"></span>[19] Harm de Vries. In the long (context) run, 2023. Accessed: 2024-06-24.
- <span id="page-20-13"></span>[20] Jerome A Feldman. Dynamic connections in neural networks. *Biological cybernetics*, 46(1):27–39, 1982.
- <span id="page-20-16"></span>[21] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In *International conference on machine learning*, pages 1126–1135. PMLR, 2017.
- <span id="page-20-10"></span>[22] A. Gammerman, V. Vovk, and V. Vapnik. Learning by transduction. In *In Uncertainty in Artificial Intelligence*, pages 148–155. Morgan Kaufmann, 1998.
- <span id="page-20-11"></span>[23] Yossi Gandelsman, Yu Sun, Xinlei Chen, and Alexei A. Efros. Test-time training with masked autoencoders. *Advances in Neural Information Processing Systems*, 2022.
- <span id="page-20-5"></span>[24] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling, 2020.
- <span id="page-20-4"></span>[25] Xinyang Geng. EasyLM: A Simple And Scalable Training Framework for Large Language Models. <https://github.com/young-geng/EasyLM>, mar 2023. [https://github.com/](https://github.com/young-geng/EasyLM) [young-geng/EasyLM](https://github.com/young-geng/EasyLM).
- <span id="page-20-15"></span>[26] Riccardo Grazzi, Julien Siems, Arber Zela, Jörg KH Franke, Frank Hutter, and Massimiliano Pontil. Unlocking state-tracking in linear rnns through negative eigenvalues. *International Conference on Learning Representations (ICLR)*, 2024.
- <span id="page-20-0"></span>[27] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023.
- <span id="page-20-12"></span>[28] Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenyà, Pieter Abbeel, Alexei A Efros, Lerrel Pinto, and Xiaolong Wang. Self-supervised policy adaptation during deployment. *arXiv preprint arXiv:2007.04309*, 2020.
- <span id="page-20-9"></span>[29] Moritz Hardt and Yu Sun. Test-time training on nearest neighbors for large language models. *arXiv preprint arXiv:2305.18466*, 2023.
- <span id="page-20-8"></span>[30] Horace He. Strangely, matrix multiplications on gpus run faster when given "predictable" data! [short], 2024. Accessed: 2024-06-30.
- <span id="page-20-2"></span>[31] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2016.
- <span id="page-20-14"></span>[32] Geoffrey E Hinton and David C Plaut. Using fast weights to deblur old memories. In *Proceedings of the ninth annual conference of the Cognitive Science Society*, pages 177–186, 1987.
- <span id="page-20-1"></span>[33] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997.
- <span id="page-20-7"></span>[34] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022.

- <span id="page-21-2"></span>[35] Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention. In *International Conference on Machine Learning*, pages 9639–9659. PMLR, 2022.
- <span id="page-21-14"></span>[36] Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. Practical computational power of linear transformers and their recurrent and self-referential extensions. *Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023.
- <span id="page-21-13"></span>[37] Kazuki Irie, Francesco Faccio, and Jürgen Schmidhuber. Neural differential equations for learning to program neural nets through continuous learning rules. *Advances in Neural Information Processing Systems*, 35:38614–38628, 2022.
- <span id="page-21-10"></span>[38] Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. *Advances in Neural Information Processing Systems*, 34:7703–7717, 2021.
- <span id="page-21-11"></span>[39] Kazuki Irie, Imanol Schlag, Róbert Csordás, and Jürgen Schmidhuber. A modern self-referential weight matrix that learns to modify itself. In *International Conference on Machine Learning*, pages 9660–9677. PMLR, 2022.
- <span id="page-21-12"></span>[40] Kazuki Irie and Jürgen Schmidhuber. Images as weight matrices: Sequential image generation through synaptic learning rules. *International Conference on Learning Representations (ICLR)*, 2022.
- <span id="page-21-6"></span>[41] Vidit Jain and Erik Learned-Miller. Online domain adaptation of a pre-trained cascade of classifiers. In *CVPR 2011*, pages 577–584. IEEE, 2011.
- <span id="page-21-5"></span>[42] Thorsten Joachims. *Learning to classify text using support vector machines*, volume 668. Springer Science & Business Media, 2002.
- <span id="page-21-0"></span>[43] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.
- <span id="page-21-1"></span>[44] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In *International conference on machine learning*, pages 5156–5165. PMLR, 2020.
- <span id="page-21-3"></span>[45] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.
- <span id="page-21-9"></span>[46] Louis Kirsch and Jürgen Schmidhuber. Meta learning backpropagation and improving it. *Advances in Neural Information Processing Systems*, 34:14122–14134, 2021.
- <span id="page-21-7"></span>[47] Ben Krause, Emmanuel Kahembwe, Iain Murray, and Steve Renals. Dynamic evaluation of neural sequence models. In *International Conference on Machine Learning*, pages 2766–2775. PMLR, 2018.
- <span id="page-21-8"></span>[48] Ben Krause, Emmanuel Kahembwe, Iain Murray, and Steve Renals. Dynamic evaluation of transformer language models. *arXiv preprint arXiv:1904.08378*, 2019.
- <span id="page-21-4"></span>[49] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th Symposium on Operating Systems Principles*, pages 611–626, 2023.
- <span id="page-21-15"></span>[50] Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. *Behavioral and brain sciences*, 40:e253, 2017.

- <span id="page-22-2"></span>[51] Quoc V Le. Building high-level features using large scale unsupervised learning. In *2013 IEEE international conference on acoustics, speech and signal processing*, pages 8595–8598. IEEE, 2013.
- <span id="page-22-7"></span>[52] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. *arXiv preprint arXiv:2402.08268*, 2024.
- <span id="page-22-9"></span>[53] Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. Consistent video depth estimation. *ACM Transactions on Graphics (ToG)*, 39(4):71–1, 2020.
- <span id="page-22-3"></span>[54] Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In *International conference on machine learning*, pages 2113–2122. PMLR, 2015.
- <span id="page-22-14"></span>[55] Luke Metz, Niru Maheswaranathan, Brian Cheung, and Jascha Sohl-Dickstein. Meta-learning update rules for unsupervised representation learning. *arXiv preprint arXiv:1804.00222*, 2018.
- <span id="page-22-8"></span>[56] Ravi Teja Mullapudi, Steven Chen, Keyi Zhang, Deva Ramanan, and Kayvon Fatahalian. Online model distillation for efficient video inference. *arXiv preprint arXiv:1812.02699*, 2018.
- <span id="page-22-6"></span>[57] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12:2825–2830, 2011.
- <span id="page-22-12"></span>[58] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for the transformer era. *arXiv preprint arXiv:2305.13048*, 2023.
- <span id="page-22-1"></span>[59] Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Teddy Ferdinan, Haowen Hou, Przemysław Kazienko, et al. Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence. *arXiv preprint arXiv:2404.05892*, 2024.
- <span id="page-22-5"></span>[60] Zhen Qin, Xiaodong Han, Weixuan Sun, Dongxu Li, Lingpeng Kong, Nick Barnes, and Yiran Zhong. The devil in linear transformer. *arXiv preprint arXiv:2210.10340*, 2022.
- <span id="page-22-4"></span>[61] Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological review*, 65(6):386, 1958.
- <span id="page-22-0"></span>[62] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In *International Conference on Machine Learning*, pages 9355–9366. PMLR, 2021.
- <span id="page-22-11"></span>[63] Imanol Schlag, Tsendsuren Munkhdalai, and Jürgen Schmidhuber. Learning associative inference using fast weight memory. *arXiv preprint arXiv:2011.07831*, 2020.
- <span id="page-22-13"></span>[64] Jürgen Schmidhuber. *Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook*. PhD thesis, Technische Universität München, 1987.
- <span id="page-22-10"></span>[65] Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. *Neural Computation*, 4(1):131–139, 1992.
- <span id="page-22-16"></span><span id="page-22-15"></span>[66] Noam Shazeer. Glu variants improve transformer, 2020.
- [67] Sam Shleifer, Jason Weston, and Myle Ott. Normformer: Improved transformer pretraining with extra normalization. *arXiv preprint arXiv:2110.09456*, 2021.

- <span id="page-23-4"></span>[68] Assaf Shocher, Nadav Cohen, and Michal Irani. "zero-shot" super-resolution using deep internal learning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 3118–3126, 2018.
- <span id="page-23-12"></span>[69] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023.
- <span id="page-23-8"></span>[70] Yu Sun, Xinhao Li, Karan Dalal, Chloe Hsu, Sanmi Koyejo, Carlos Guestrin, Xiaolong Wang, Tatsunori Hashimoto, and Xinlei Chen. Learning to (learn at test time). *arXiv preprint arXiv:2310.13807*, 2023.
- <span id="page-23-7"></span>[71] Yu Sun, Wyatt L Ubellacker, Wen-Loong Ma, Xiang Zhang, Changhao Wang, Noel V Csomay-Shanklin, Masayoshi Tomizuka, Koushil Sreenath, and Aaron D Ames. Online learning of unknown dynamics for model-based controllers in legged locomotion. *IEEE Robotics and Automation Letters*, 6(4):8442–8449, 2021.
- <span id="page-23-5"></span>[72] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In *International Conference on Machine Learning*, pages 9229–9248. PMLR, 2020.
- <span id="page-23-11"></span>[73] Sebastian Thrun and Lorien Pratt. Learning to learn: Introduction and overview. In *Learning to learn*, pages 3–17. Springer, 1998.
- <span id="page-23-9"></span>[74] Tijmen Tieleman and Geoffrey Hinton. Using fast weights to improve persistent contrastive divergence. In *Proceedings of the 26th annual international conference on machine learning*, pages 1033–1040, 2009.
- <span id="page-23-1"></span>[75] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.
- <span id="page-23-3"></span>[76] Vladimir Vapnik. *The nature of statistical learning theory*. Springer science & business media, 2013.
- <span id="page-23-0"></span>[77] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In *ICML*, page 1096–1103, 2008.
- <span id="page-23-10"></span>[78] Christoph Von Der Malsburg. The correlation theory of brain function. In *Models of neural networks: Temporal aspects of coding and information processing in biological systems*, pages 95–119. Springer, 1994.
- <span id="page-23-6"></span>[79] Renhao Wang, Yu Sun, Yossi Gandelsman, Xinlei Chen, Alexei A Efros, and Xiaolong Wang. Test-time training on video streams. *arXiv preprint arXiv:2307.05014*, 2023.
- <span id="page-23-2"></span>[80] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*, 2019.

- <span id="page-24-2"></span>[81] Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, Madian Khabsa, Han Fang, Yashar Mehdad, Sharan Narang, Kshitiz Malik, Angela Fan, Shruti Bhosale, Sergey Edunov, Mike Lewis, Sinong Wang, and Hao Ma. Effective long-context scaling of foundation models, 2023.
- <span id="page-24-1"></span>[82] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. *arXiv preprint arXiv:2312.06635*, 2023.
- <span id="page-24-0"></span>[83] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. *arXiv preprint arXiv:2406.06484*, 2024.
- <span id="page-24-4"></span>[84] Biao Zhang and Rico Sennrich. Root mean square layer normalization, 2019.
- <span id="page-24-3"></span>[85] Hao Zhang, Alexander C Berg, Michael Maire, and Jitendra Malik. Svm-knn: Discriminative nearest neighbor classification for visual category recognition. In *2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)*, volume 2, pages 2126–2136. IEEE, 2006.

### <span id="page-25-0"></span>A Dual Form

Here we derive the dual form for general MLPs of arbitrary depth, with nonlinear activations.

Without loss of generality, consider  $\eta = 1$  for convenience, and consider only the first mini-batch, where t = 1,...,b. Denote:

$$\hat{x}_t = \theta_K x_t, \quad y_t = \theta_V x_t, \quad \bar{x}_t = \theta_O x_t.$$

Also denote  $\hat{X} = [\hat{x}_1, ..., \hat{x}_b]$ , and Y and  $\bar{X}$  analogously. In general, uppercase letters denote matrices whose columns are vectors denoted by the corresponding lowercase letter.

For a network with K layers, denote the initial parameters in layer k by  $W_0^k$ . Our convention is to use superscripts for the layer and subscripts for time.

### A.1 Forward pass

During the initial forward pass of TTT, we denote the input to layer k by  $\hat{X}^k = [\hat{x}_1^k, ..., \hat{x}_b^k]$ , with  $\hat{X}^1 = \hat{X}$ . Now we write the forward pass of TTT using these notations.

For k = 1, ..., K:

- $Z^k = W_0^k \hat{X}^k$
- $\hat{X}^{k+1} = \sigma_k(Z^k)$

where  $\sigma_k$  for k = 1, ..., K can be any element-wise operation  $(\mathbb{R} \mapsto \mathbb{R})$  with derivative  $\sigma'$ .

Given  $\hat{X}^{K+1}$ , we compute the loss:

$$l = \frac{1}{2} \ell \left( W_0^1, \dots, W_0^K; X \right) = \frac{1}{2} \| \hat{X}^{K+1} - Y \|_F^2 = \sum_{t=1}^b l_t,$$

where  $l_t = \frac{1}{2} ||\hat{x}_t^K - y_t||^2$  is the same as defined in Equation 4, except scaled by 1/2 for convenience.

All the operations above (except  $\sigma$ ) are matmuls and sums, therefore are hardware efficient. Both the primal form and the dual form share these initial operations.

#### A.2 Primal form

The primal form first computes  $G_t^k = \nabla_{W_0^k} l_t$  for t = 1, ..., b, then updates  $W_t^k = W_0^k - \sum_{s=1}^t G_s^k$ . Finally, given  $\bar{X}^1 = [\bar{x}_1^1, ..., \bar{x}_h^1] = \bar{X}$ , the primal form repeats the forward pass with the updated  $W_s$ .

For k = 1, ..., K:

- $\bar{z}_t^k = W_t^k \bar{x}_t^k$ , for t = 1, ..., T
- $\bar{x}_t^{k+1} = \sigma_k(\bar{z}_t^k)$ , for t = 1, ..., T

where  $\bar{X}^{K+1} = [\bar{x}_1^{k+1}, \dots, \bar{x}_b^{k+1}]$  contains the output tokens.

Note that a standard backward pass only computes the sum of the gradients:

$$\nabla_{W_0^k} l = \sum_{t=1}^b \nabla_{W_0^k} l_t = \sum_{t=1}^b G_t^k,$$

so the computation of the individual terms in the sum  $G_t^k$  for t = 1, ..., b cannot be batched together into matmuls. Similarly, the forward pass in primal form uses a different  $W_t$  for each  $\bar{x}_t$ , therefore also cannot be batched in the same way as a standard forward pass. These non-standard passes have poor hardware efficiency.

#### A.3 Dual form

As discussed in Subsection 2.5, the goal of the dual form is to compute  $\bar{X}^{K+1}$  and  $W_b^1, \ldots, W_b^K$  with only matmuls and light-weight operations such as sums,  $\sigma$ , and  $\sigma'$ . To achieve this goal, we avoid explicitly computing the intermediate variables:  $G_t^k$  and  $W_t^k$  for  $t = 1, \ldots, b$ .

The dual form first computes  $\nabla_{\hat{X}^{K+1}}l = \hat{X}^{K+1} - Y$ , then takes a standard backward pass.

For k = K, ..., 1:

- $\nabla_{Z^k} l = \sigma'_k(Z^k) \odot \nabla_{\hat{X}^{k+1}} l$
- $\nabla_{\hat{X}^k} l = \left(W_0^k\right)^T \nabla_{Z^k} l$
- $\nabla_{W_0^k} l = \nabla_{Z^k} l \left( \hat{X}^k \right)^T$

where  $\sigma'$  is applied element-wise, and  $\odot$  is element-wise multiplication.

Now we can already compute  $W_b^k = W_0^k - \nabla_{W_0^k} l$ . To compute the output tokens, we do another forward pass.

For k = 1, ..., K:

- $\bar{Z}^k = W_0^k \bar{X}^k \nabla_{Z^k} l \cdot \mathsf{mask} \left( \left( \hat{X}^k \right)^T \bar{X}^k \right)$
- $\bar{X}^{k+1} = \sigma(\bar{Z}^k)$

By the end of the forward pass, we have computed  $\bar{X}^{K+1}$ .

While this forward pass is non-standard, it only contains matmuls, sums,  $\sigma$ , and mask, therefore is efficient like the standard forward pass.

#### A.4 Derivation

To derive the dual form, we show that:

$$\bar{Z}^k = W_0^k \bar{X}^k - \nabla_{Z^k} l \cdot \mathsf{mask} \left( \left( \hat{X}^k \right)^T \bar{X}^k \right)$$

is the same as what would be computed in the primal form. Specifically, we show that each column  $\bar{z}_t^k$  of  $\bar{Z}^k$  in the second forward pass of the dual equals to  $W_t^k \bar{x}_t^k$  in the forward pass of the primal. We invoke a simple fact.

**Fact 1.** Define matrices  $A = [a_1, ..., a_b]$ ,  $Q = [q_1, ..., q_b]$ , and  $V = [v_1, ..., v_b]$ . <sup>14</sup> Define  $\hat{v}_t = \sum_{s=1}^t a_s^T q_t v_s$ , and  $\hat{V} = [\hat{v}_1, ..., \hat{v}_b]$ , then  $\hat{V} = V \cdot \mathsf{mask}(A^T Q)$ .

Now plug  $A = \hat{X}^k$ ,  $Q = \bar{X}^k$ ,  $V = \nabla_{Z^k} l$ , and  $\hat{V} = W^k \bar{X}^k - \bar{Z}^k$  into the fact above, we have shown the desired equality.

Note that the  $\sigma_k$  and  $\sigma_k'$  used above can be extended to arbitrary functions that are not necessarily element-wise operations, including normalization layers. This extension can be achieved through, for example, vjp (vector-Jacobian product) in standard libraries for automatic differentiation such as JAX and PyTorch. However, the dual form cannot accelerate operations inside  $\sigma$  or its vjp.

 $<sup>^{14}</sup>$ Our matrix A would usually be denoted by K in another context. We use A to avoid confusion with the layer number K.

## <span id="page-27-0"></span>**B** Nadaraya-Watson estimator

**Derivation for the Nadaraya-Watson estimator.** Throughout this section, we use  $\mathbf{x}$  to denote the input token x as a random variable. Our desired output is the corresponding output token, another random variable  $\mathbf{z}$ . This is formulated as estimating the conditional expectation of  $\mathbf{z}$ :

$$\mathbb{E}[\mathbf{z}|\mathbf{x}=x] = \int p(z|x) \ z \ dz = \int \frac{p(x,z)}{p(x)} \ z \ dz.$$

Since the true probability distributions p(x) and p(x,z) are unknown, we replace them with their kernel density estimations. Specifically, the kernel density estimation for p(x) is:

$$\hat{p}(x) = \frac{1}{n} \sum_{i=1}^{n} \kappa(x, x_i),$$

where each  $x_i$  is a piece of training data in general. (Recall that for our paper,  $x_i$  is specifically training data for the inner loop, *i.e.* a token, which matches our notation in the main text.)

For estimating p(x, y), we use the product kernel:

$$\hat{p}(x,z) = \frac{1}{n} \sum_{i=1}^{n} \kappa(x,x_i) \kappa'(z,z_i).$$

At first sight, it seems absurd to factor the joint probability into two seemingly independent kernels. But in this case,  $\kappa'$  can actually be any  $\kappa'_i$  dependent on  $x_i$ , since it will be integrated out. So the two kernels do not need to be independent.

Plugging in those estimations, we obtain the Nadaraya-Watson estimator:

$$\hat{\mathbb{E}}[\mathbf{z}|\mathbf{x}=x] = \int \frac{\hat{p}(x,z)}{\hat{p}(x)} z \, dz$$

$$= \frac{1}{\hat{p}(x)} \int \hat{p}(x,z) z \, dz$$

$$= \frac{1}{\sum_{i=1}^{n} \kappa(x,x_i)} \int \sum_{i=1}^{n} \kappa(x,x_i) \kappa'(z,z_i) z \, dz$$

$$= \frac{1}{\sum_{i=1}^{n} \kappa(x,x_i)} \sum_{i=1}^{n} \kappa(x,x_i) \int \kappa'(z,z_i) z \, dz$$

$$= \frac{1}{\sum_{i=1}^{n} \kappa(x,x_i)} \sum_{i=1}^{n} \kappa(x,x_i) z_i.$$

**Asymmetric kernels.** In modern days, people think of kernels as positive semi-definite, which might not be guaranteed for  $\kappa$  unless  $\theta_K = \theta_Q$ . However, people working on kernels decades ago, around the time when the Nadaraya-Watson estimator was popular, have been very lenient with the choice of kernels, and asymmetric kernels such as our  $\kappa$  in Equation 10 have enjoyed a long tradition: When a kernel estimator uses  $\theta_K \neq \theta_Q$ , it is known as a balloon estimator [14]. Papers such as Breiman et al. [10] have even used  $\theta_Q$  as a function of x', known as sample-adaptive smoothing.

<span id="page-28-0"></span>![](_page_28_Figure_0.jpeg)

Figure 13. Left: A residual block, the basic building block for Transformers. The sequence modeling block is instantiated into two variants: the Transformer backbone and Mamba backbone. Middle: TTT layer in the Transformer backbone. The LN before *O* comes from NormFormer [\[67\]](#page-22-15). Right: TTT layer in the backbone inspired by Mamba [\[27\]](#page-20-0) and Griffin [\[18\]](#page-20-3). Following these two architectures, *σ* here is GELU [\[31\]](#page-20-2). To accommodate the extra parameters of the gate without changing the embedding dimension, we simply combine *θK* and *θQ* into a single projection.

# <span id="page-28-1"></span>C Experiment details

Architectures. Our Transformer strictly follows the construction in the Mamba paper, where *Transformer* is called *Transformer++*. Specifically, the Transformer architecture is based on Llama [\[75\]](#page-23-1), with rotary positional encodings (RoPE) [\[69\]](#page-23-12), SwiGLU MLP blocks [\[66\]](#page-22-16), and RMSNorm [\[84\]](#page-24-4) instead of LayerNorm. Our Mamba baseline uses the public code provided by the authors. We have verified that our baselines can reproduce the numbers reported in [\[27\]](#page-20-0).

Training configurations. Our training configurations are in Table [3,](#page-29-0) which simply reproduces Table 12 in the Mamba paper. As discussed in Footnote [12,](#page-13-0) all models are trained with a batch size of 0.5M tokens regardless of context length. All of our optimization hyper-parameters follow the "improved recipe" in Appendix E.2 of the Mamba paper, reproduced below:

- AdamW optimizer: *β* = (0*.*9*,*0*.*95)
- Cosine schedule: decay to end learning rate 1*e* − 5
- Linear learning rate warmup over 10% of the training steps
- Weight decay: 0.1
- Gradient clipping: 1.0
- No Dropout
- Mixed Precision

<span id="page-29-0"></span>

| Params. | Blocks | Embed. dim. | Heads | Train steps | Peak LR | Tokens |
|---------|--------|-------------|-------|-------------|---------|--------|
| 125M    | 12     | 768         | 12    | 4800        | 3e-3    | 2.5B   |
| 350M    | 24     | 1024        | 16    | 13500       | 1.5e-3  | 7B     |
| 760M    | 24     | 1536        | 16    | 29000       | 1.25e-3 | 15B    |
| 1.3B    | 24     | 2048        | 32    | 50000       | 1e-3    | 26B    |

Table 3. Training configurations for all experiments. This table reproduces Table 12 in the Mamba paper. The only difference is that the learning rate they use for Mamba and Transformer is 5× the values in their Table 12, and we report the actual values (5×). Note that this table only applies to TTT-Linear, TTT-MLP, and Transformers, as Mamba does not follow the multi-head residual block structure inherited from Transformers.

As discussed in Footnote 10, all models are trained using the Llama tokenizer [75]. For experiments on the Pile, this is the only difference with the recipe in the Mamba paper, which uses two other tokenizers. For experiments on Books, we find that the original angle of the RoPE encoding [69]  $\theta = 10,000$  is sub-optimal for our Transformer baseline in long context. Starting at context length 4k, we try  $\theta = 500,000$  following the Llama Long paper [81], and use the better perplexity for Transformer (both pretrain and finetune).

**Transformer finetuning.** Finetuning starts a new cosine schedule with the same optimization hyper-parameter as training from scratch, except the peak learning rate. We try three peak learning rates for finetuning: 1e-5, 1e-4, and 1e-3, and select for the best perplexity. We observe that 1e-4 works the best for the 125M models, while 1e-5 works the best for 350M and larger. This observation is reasonable considering that the end learning rate for the Chinchilla recipe is 1e-5.

**Learning rate for TTT.** As mentioned in Subsection 2.7, the inner-loop base learning rate  $\eta_{\text{base}}$  is set to 1 for TTT-Linear and 0.1 for TTT-MLP. Our heuristic for setting  $\eta_{\text{base}}$  is similar to how people set the outer-loop learning rate for regular training: We tried  $\eta_{\text{base}} \in \{0.01, 0.1, 1, 10\}$  and used the largest value that does not cause instabilities. For TTT-MLP, we use linear warmup for  $\eta_{\text{base}}$  over 10% of the training steps, similar to regular training. The number of training steps in the inner loop is T/b (assume divisible). For TTT-Linear, we tried linear warmup in the inner loop but did not observe a difference.

**Experiments in Figure 2 (right).** To ensure fairness to Mamba, all methods in these experiments have matched training FLOPs and are trained with the same recipe (last row of Table 3) as Mamba 1.4B. For TTT-Linear and TTT-MLP, matched training FLOPs also imply matched inference FLOPs. Transformer (TF finetune) has 2.8× the inference FLOPs, giving it an advantage as our baseline. To match training FLOPs with Mamba, Transformer has 19 blocks instead of 24. For TTT-Linear and TTT-MLP, their training FLOPs are already close to those of Mamba, so we only need to change the hidden dimension of the MLP blocks from 5504 to 5808 for TTT-Linear and 5248 for TTT-MLP.

**Gradient checkpointing through time.** By default, libraries such as JAX and PyTorch save the intermediate activations during a forward pass so they can be reused during the backward pass. However, for a TTT layer with W as hidden state, this default saves  $W_1, \ldots, W_T$ , which uses too much memory. With TTT mini-batch and the dual form, we still need to save (assume divisible)  $\kappa = T/b$  Ws at the end of the mini-batches. A standard technique to save memory in this scenario is gradient checkpointing [12], which is usually applied through layers, but we apply it through time.

<span id="page-30-0"></span>![](_page_30_Figure_0.jpeg)

Figure 14. The self-supervised TTT loss  $\ell$  averaged over all test sequences of the form  $x_1, \ldots, x_T$  where T=2048, for all 12 TTT layers in a network with 125M parameters train on the Pile. The same network is also used for b=1 (online GD) in the left panel of Figure 7. For layers in the middle, we observe that  $\|x_t\|$  rises steadily, causing all three losses to rise with it. Even for these layers, the gap between  $\ell(W_0; x_t)$  and  $\ell(W_t; x_t)$  still increases with t. For visual clarity, loss values have been averaged over a sliding window of 10 timesteps.

<span id="page-31-0"></span>![](_page_31_Figure_0.jpeg)

Figure 15. Complete results on Books, presented by context lengths. Figure 11 in Subsection 3.2 presents the subset of results for context lengths 2k and 32k.

<span id="page-32-0"></span>![](_page_32_Figure_0.jpeg)

Figure 16. An alternative view of our complete results on Books, presented by model sizes, with context length as the x-axis. For all methods trained from scratch, perplexity becomes worse once the context length becomes too large. This trend is not observed with TF finetune, except for one case at the 125M scale. The best context length increases for larger models (trained from scratch).