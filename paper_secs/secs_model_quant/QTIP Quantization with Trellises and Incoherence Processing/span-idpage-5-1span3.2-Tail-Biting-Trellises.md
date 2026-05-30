# <span id="page-5-1"></span>3.2 Tail-Biting Trellises

Directly quantizing a length-T sequence to a (L, k, V ) trellis results in a total of kT + L − kV bits since the starting state takes an additional L − kV bits to store. If we run inference on a machine with w-bit words where w|kT, we must read an extra ⌈ L−kV w ⌉w − (L − kV ) wasted bits per sequence. For common w (e.g. 32), setting L = kV + w makes the Viterbi algorithm intractable. One way to solve this is by enforcing that the start and end state share L − kV bits, i.e. the trellis is *tail-biting* [\[4\]](#page-11-11). Exactly solving the tail-biting trellis problem via dynamic programming takes time quadratic in the state space (2 <sup>L</sup>), making this problem intractable for reasonable L ≥ 12 [\[29\]](#page-12-12). However, since RHT-processed weights are approximately i.i.d., simple algorithms can be effective for approximately solving the tail-biting problem. We propose Algorithm [4,](#page-7-0) which first rotates the sequence by T /2, quantizes it, and then extracts the overlap between the rotated start and end states. It then requantizes the original sequence with this overlap as the tail-biting overlap. This only requires two Viterbi calls

### Algorithm 2 Computed Gaussian Code "3INST"

```
input L-bit 0 left-padded integer x, uint32 a,b, float16 m. x \leftarrow (ax+b) \mod 2^{32} {run LCG to get uniform random x} {modify sign, mantissa, and bottom 2 exponent bits of m and sum, this is approximately Gaussian} m \leftarrow \texttt{reinterpret}(m, \texttt{uint32}) << 16 + \texttt{reinterpret}(m, \texttt{uint32}) x \leftarrow (x & \texttt{b}1000111111111111111111111111111111111
```

