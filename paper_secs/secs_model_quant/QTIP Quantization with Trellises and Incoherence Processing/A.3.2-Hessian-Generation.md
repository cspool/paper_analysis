# A.3.2 Hessian Generation

Hessian matrices were generated with 6144 sequences of length 2048 for Llama 1, 6144 sequences of length 2048 for Llama 2, 4096 sequences of 8192 for Llama 3, and 4096 sequences of 8192 for Llama 3.1 except for 405B, which only used 2048 sequences due to time constraints. All sequences were sampled from the RedPajama dataset [\[7\]](#page-11-14).

### Algorithm 5 QTIP with BlockLDLQ

```
 \begin{split} & \text{input } W \in \mathbb{R}^{m \times n}, H \in \mathbb{R}^{n \times n}, T_x, T_y, L, k, V, \operatorname{code} C. \\ & \hat{W} \leftarrow 0_{m,n} \\ & LDL^T \leftarrow T_y\text{-block LDL decomposition of } H \\ & A \leftarrow L - I \\ & \text{for } j \in \{n/T_y - 1, n/T_y - 2, ..., 0\} \text{ do} \\ & x \leftarrow W_{:,j}T_y:(j+1)T_y + (W_{:,j}T_y: -\hat{W}_{:,j}T_y:)A_{j}T_y:(j+1)T_y} \\ & x \leftarrow x.\operatorname{reshape}(m/T_x, T_xT_y) \\ & \hat{x} \leftarrow \operatorname{Viterbi}(x, (L, k, V) \text{ bitshift trellis, } C) \text{ (row-wise)} \\ & \hat{W}_{:,j}T_y:(j+1)T_y \leftarrow \hat{x}.\operatorname{reshape}(m, T_y) \\ & \text{end for} \\ & \text{output Quantized } \hat{W}. \end{split}
```

#### A.3.3 Hadamard Matrices

We use Hadamard matrices from Neil Sloane's website [30].

### A.3.4 Perplexity

We use the OPTQ "Wikitext2" and "C4" test splits to calculate perplexity [14].

#### A.3.5 Fine Tuning

We use the same fine-tuning hyperparemters as QuIP#, except that we do not perform matrix fusion [34]. This results in having to quantize 7 matrices (q, k, v, o, up, gate, down) instead of 4 (qkv, o, upgate, down). We quantize and fine tune in the following order: (v, q, k, o, up gate, down).

### A.4 Example Generations

The following are some sample generations from Llama 2 70B (not instruction tuned) quantized to 2 bits with QTIP, truncated to 256 tokens.

**Prompt:** It is a truth universally acknowledged

QTIP: It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding residents, that he is considered the rightful property of some one or other of their daughters. "My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?" "But it is," returned she; "for Mrs. Long has just been here, and she told me that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and that he is now in town in search of a wife." "I see no occasion for that. You and the girls may go, or you may send them by themselves, which do you prefer? Only let me know what you are meaning to do, for I shall certainly not be at home. You

**Prompt:** Call me Ishamel

QTIP: Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen, and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypochondriacal palate is out of sorts, and I am desperately dull, and in the dumps, and have such an overpowering sense of fatality that I find myself involuntarily gravitating toward Cemeteries—and especially toward those open ones which are occasionally cheered by the gleam of a bonfire, or a cigar, or a pipe, or a lantern, or a candle, or a torch, or the moon, I then take to the ship. There is nothing surprising in this,

The following is a sample generation from Llama 3.1 405B Instruct quantized to 2 bits with QTIP.

**Prompt:** Give a proof for why the square root of 2 is irrational:

**QTIP:** The proof that the square root of 2 is irrational is as follows: Assume, for the sake of contradiction, that  $\sqrt{2}$  is rational. This means it can be expressed as the ratio of two integers, a and b, where b is non-zero:

$$\sqrt{2} = a/b$$

We can further assume that a and b are not both even (i.e., they do not share a common factor of 2), because if they were, we could divide both by 2 and obtain a smaller representation of  $\sqrt{2}$ . Squaring both sides of the equation, we get:

$$2 = a^2/b^2$$

Multiplying both sides by  $b^2$ , we get:

$$2b^2 = a^2$$

This means that  $a^2$  is even, because it is equal to 2 times an integer  $(b^2)$ . Therefore, a must also be even, because the square of an odd number is always odd. Since a is even, we can write a=2c, where c is an integer. Substituting this into the equation above, we get:

$$2b^2 = (2c)^2$$

Expanding the right-hand side, we get:

$$2b^2 = 4c^2$$

Dividing both sides by 2, we get:

$$b^2 = 2c^2$$

This means that  $b^2$  is even, and therefore b is also even. However, this contradicts our assumption that a and b are not both even. Therefore, our original assumption that  $\sqrt{2}$  is rational must be false, and  $\sqrt{2}$  is actually irrational. This proof is often attributed to the ancient Greek mathematician Euclid, and it is still considered one of the most elegant and simple proofs of the irrationality of  $\sqrt{2}$ .

