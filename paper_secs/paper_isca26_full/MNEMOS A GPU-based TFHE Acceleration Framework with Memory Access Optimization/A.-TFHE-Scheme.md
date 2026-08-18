# A. TFHE Scheme

Torus-based Fully Homomorphic Encryption (TFHE) supports arbitrary computation over encrypted data (ciphertext) in both Boolean and integer domains. It enables direct evaluation of logical operations, such as comparisons and bitwise functions, as well as exact high-precision arithmetic [4, 18]. With its bit-level expressiveness and precise computation semantics, TFHE is well suited for applications requiring deterministic and accurate processing, such as quantized neural network inference, where encrypted data can be evaluated without approximation [35].

The cornerstone of TFHE is its programmable bootstrapping (PBS), a fundamental operation that simultaneously refreshes ciphertexts and evaluates arbitrary functions on ciphertext [4]. Algorithm 1 presents the overview of PBS. During PBS, the accumulated noise within a ciphertext is reduced while a user-defined function f(m) is homomorphically applied to the underlying plaintext m. The process consists of four sequential stages: Modulus Switching  $\rightarrow$  Blind Rotation  $\rightarrow$  Sample Extraction  $\rightarrow$  Key Switching.

The input of programmable bootstrapping is an LWE ciphertext consisting of (n+1) scalar elements, denoted as  $\mathbf{c}=(a_0,a_1,a_2,\ldots,a_{n-1},b)$ . In the first stage, the Modulus Switching operation rescales and rounds each component of the ciphertext from modulus p to 2N, such that  $\tilde{a}i=\lfloor 2Na_i \rceil 2N$  and  $\tilde{b}=\lfloor 2Nb \rceil_{2N}$ . This transformation maps the ciphertext into the torus domain, thereby enabling subsequent torus-based homomorphic operations.

Next, the rounded ciphertext undergoes a Blind Rotation operation, which leverages the key-dependent

