# B.5 Long Context Models

Long context has become a popular subject, and several recent models have claimed to scale to longer and longer sequences. However, these are often from a computational standpoint and have not been extensively validated. These include:

- Recurrent Memory Transformer (Bulatov, Kuratov, and Burtsev [2023\)](#page-17-17), a lightweight wrapper around a Transformer backbone. It showed ability to generalize up to 1M sequences but only on synthetic memorization tasks; their main result is similar to our Induction Heads extrapolation experiment (Table [2\)](#page-10-2).
- LongNet (Ding et al. [2023\)](#page-17-18), which claimed to scale to 1B length but only evaluated on length < 100 for actual tasks.
- Hyena and HyenaDNA (Nguyen, Poli, et al. [2023;](#page-20-8) Poli et al. [2023\)](#page-20-1), which claimed to leverage up to 1M context. However, their experiments trained on proportionally more data at longer contexts, making it hard to conclude if quality improvements at 1M context are due to context length or due to more data and computation.
- Sparse Transformer (Child et al. [2019\)](#page-17-19) showed a proof-of-concept of using a strided sparse attention Transformer to model audio waveforms of length 2 <sup>20</sup> = 1048576, although did not discuss performance tradeoffs when controlling for computation and model size.

In contrast, we believe this work presents one of the first approaches to meaningfully demonstrate increasing performance with longer context.

