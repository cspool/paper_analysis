# *b) Potential Target Bit Flip Attacks with* m1*:*

If m<sup>1</sup> overlaps with the pointer ptr, and X denote the bitmask for the overlapping bits. Then the attacker can construct (m1,a = m1,v ⊕X, p<sup>a</sup> = p<sup>v</sup> ⊕X, c<sup>a</sup> = cv) where (mv, pv, cv) is a valid tuple. As a result, the attacker can overwrite the modifier to flip certain bits in the pointer if X is not zero.

To avoid this ambiguity, m<sup>1</sup> should be placed only in pointer bits that do not affect address generation. On most 64-bit architectures, only the lower A bits are used as virtual-address bits, e.g., A = 48 for a 48-bit virtual address space; the remaining 64−A high-order bits are unused or sign-extension bits. In addition, if pointers are word-aligned, the two leastsignificant bits do not affect the addressed word. These bits can therefore be safely used for m1, yielding

$$|m_1|_{\text{max}} = 64 - A + 2.$$

- *c) Potential Key-Collision Attacks with* m2*:* Assume the operating system assigns the victim process a secret key k<sup>v</sup> and the attacker process a key ka. A potential concern is whether an attacker could exploit the m<sup>2</sup> modifier to induce a key collision across protection domains. Specifically, suppose the attacker attempts to construct a relation of the form k<sup>a</sup> = k<sup>v</sup> ⊕ m2,d where m2,d denotes a chosen modifier difference. If such a relation were achievable, the attacker could attempt to craft a tuple (m2,a = m2,d ⊕ m2,v, p<sup>a</sup> = pv, c<sup>a</sup> = Enck<sup>a</sup> (pv)), and inject it into the victim domain, thereby forging a valid pointer. We prevent such a collision with our key assignment.
- *d) Key Assignment:* To prevent cross-domain keycollision attacks induced by adversarial choices of m2, the system enforces that the *unaffected* portion of the domain key is unique across security domains. Concretely, if two domains share the same 128−|m2| unaffected key bits, then an attacker could choose m<sup>2</sup> values that cause their derived key to match the victim's derived key. Ensuring uniqueness of the unaffected portion eliminates this possibility. This policy also bounds the number of simultaneously supported security domains: since only 128 − |m2| bits are reserved for domain separation, the maximum number of unique domains is at most 2 128−|m2| .

#### *e) Proof of Security:*

We define AdvLIP P EN (A) denote the advantage of adversary A in forging an encrypted pointer c<sup>a</sup> for chosen (ma, pa) for LIPPEN knowing a set of tuples T , and AdvE(B) being the advantage of adversary B in distinguishing the block cipher EK(·) from a uniformly random permutation P even with oracle access to both EncEK(·) and DecEK(·). For LIPPEN, the implementation ensures that |m1| uses only unused pointer bits, and the 128 − |m2| key bits are never shared between domains.

Theorem 1. *For any probabilistic polynomial-time (PPT) adversary* A *knowing* T *running in time* t*, there exists a PPT adversary* B *such that*

$$\operatorname{Adv}_{\operatorname{Lippen}}(\mathcal{A}) \leq \operatorname{Adv}_{E}(\mathcal{B}) + \varepsilon(q).$$

*Proof.* B is given oracle access to either a real block cipher E(·) or a random permutation P for both encryption and decryption. B runs A by simulating the protocol using its oracle in place of E(·) as follows.

- 1) Answer all of A's queries to build T by querying the oracle, i.e., encryption or decryption to obtain c<sup>i</sup> with key k<sup>v</sup> ⊕ m2,i and plaintext p<sup>i</sup> ⊕ m1,i.
- 2) For (ma, pa), run A to forge pointer ca. B queries the oracle to encrypt plain = p<sup>a</sup> ⊕m1,a with key k<sup>v</sup> ⊕m2,a. B repeats for q times and if more than q/2 encryption of plain matches ca, B returns that a real block cipher E(·) is behind the oracle; otherwise, B returns that a random permutation P is behind the oracle.

If the oracle is for a real block cipher, then the probability of B returning the right value (i.e., AdvE(B)) is the probability A returning the right c<sup>a</sup> more than half of the time (i.e., Adv<sup>L</sup>IPPEN(A)). If the oracle is for a random permutation P, then B will return the wrong result if c<sup>a</sup> happens to be the output of the random permutation, which is of probability 2 −64×q/2 . Thus, AdvE(B) ≥ AdvLIP P EN (A) − ε(q)

Thus, breaking LIPPEN is not easier than breaking the underlying block cipher. The security of the scheme therefore reduces to the cryptographic strength of the underlying PRINCE-family cipher. Existing cryptanalysis of the PRINCE family primarily targets reduced-round variants or relies on implementation attacks such as differential fault analysis [3], [35], [51], [69], [86], [87] and has not produced practical attacks on the full-round constructions, indicating no practical key-recovery; the best known attacks require significantly reduced-round variants or complexity close to exhaustive search. More specifically, the best cryptanalysis efforts in PRINCE (base design for PRINCEv2) report that a single key can be recovered with a computational complexity of 2 125.47 using structural linear relations; in the related key setting, the memory complexity is 2 <sup>33</sup> and the time complexity 2 <sup>64</sup>; using the related key boomerang attack, the complexity is 2 <sup>39</sup> for both memory and time [51]. The authors of PRINCEv2 claim that there is no attack against PRINCEv2 with memory complexity below 2 <sup>47</sup> (chosen) plaintext-ciphertext pairs (obtained under the same key) and time-complexity below 2 <sup>112</sup> [24], which is in line with the NIST requirement on the security of lightweight ciphers [71].

TABLE V: Modifier in representative PAC-based defenses.

| Work                       | Modifier Design                                                                                                                              |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| PARTS [62]                 | Stack pointer (SP) for return addresses,<br>and a type identifier for indirect and<br>data pointers.                                         |
| PACStack [61]              | Previous return address on the stack.                                                                                                        |
| PTAuth [40]                | A generated object-id.                                                                                                                       |
| PACSan / PACMem [59], [60] | A static random number generated at<br>compile time.                                                                                         |
| AOS [53]                   | Stack pointer (SP) for return addresses.                                                                                                     |
| PACTight [49]              | Pointer location and a random tag for<br>sensitive data pointers; previous return<br>address and unique function ID for<br>return addresses. |
| RSTI [48]                  | Unique mixture of pointer scope, type,<br>permission, and location information.                                                              |

*4) Integrity with Encryption:* Theorem 1 shows that the probability of an adversary successfully forging a pointer for a target address and modifier is no larger than breaking the cipher. For a random encrypted pointer, after decryption, if the unused pointer bits do not match m1, the engine will detect that the encrypted pointer is not valid with detection probability of 1 − 2 −|m1| , which is at the same security level as PAC. Even if the modifier matches, the pointer will point to a random place in the address space not controlled by the attacker, while with PAC the attacker can forge a pointer to an attacker-chosen address.

*5) Number of Modifier Bits Needed in Practice:* Given that the security will depend on the number of bits in the modifier, here we study the entropy needed in the modifier in practice.

Different pointer-protection schemes employ different modifier types, as summarized in Table V. Apple's PAC [6] uses a zero modifier for function and vtable pointers, requiring only a single unique value In general, the number of modifier bits required is proportional to the number of distinct modifier values that must be represented.

PARTS-CFI [62] uses a 64-bit type identifier, but the effective entropy needed depends on the number of distinct pointer types and variables in the program (e.g., int\*, char\*, etc.) For example, in xalancbmk, one of the largest benchmarks in both SPEC CPU2006 and SPEC CPU2017, there are 2,558 pointer types and 32,097 pointer variables [48], corresponding to roughly 12 bits of modifier space to ensure uniqueness. The same reasoning extends to other PAC-based defenses. RSTI [48] derives modifiers from pointer scope, type, permission, and location, increasing the unique context count to 14,073 for xalancbmk and requiring approximately 14 bits; assigning a distinct modifier per pointer variable would require 16 bits. Since this benchmark represents the largest pointer footprint across the SPEC suites, we conclude that a 16-bit modifier space is sufficient for realistic workloads.

Since the use of m<sup>2</sup> reduces the maximum number of unique domains, once we decide the |m| based on the required entropy for context, we use all |m1|max for m1, and the remaining entropy is assigned to m2.

*F. Memory Tagging and Address Width Scaling*

Modern architectural trends, such as Memory Tagging Extensions (MTE) and the expansion of Virtual Address (VA) widths, significantly constrain the available non-canonical bits within a 64-bit pointer. Memory Tagging associates memory regions with metadata tags stored in the upper pointer bits to detect spatial and temporal violations. Simultaneously, scaling the address width directly reduces the unused bits previously available for in-pointer security metadata.

LIPPEN is designed to be agnostic to these architectural shifts. Since our encryption operates transparently on the pointer value, it preserves the integrity of any bits reserved by hardware for addressing or tagging. However, as the address space A grows or the tag field |tags| expands, the bitbudget for modifier m<sup>1</sup> is proportionally reduced. To maintain a constant security margin, LIPPEN can meet the entropy requirements by using secondary modifier m2.

Under these constraints, the number of supported distinct security domains is bounded by the remaining key-separation entropy. Specifically, the effective domain-separation space becomes E = 128−|m|−|T ag|+(64−A). For example, with |m| = 16 and |T ag| = 4: (i) if A = 48, then E = 124 bits (2 <sup>124</sup> domains); (ii) if A = 57 (x86 64 servers with 5-level page tables), then E = 115 bits (2 <sup>115</sup> domains).

#### *G. Discussion on Speculative Execution and Performance*

Pointer encryption serializes decryption with pointer dereference, introducing non-zero latency overhead on every protected access. PARTS [62] quantifies this at 4 cycles per dereference, yielding < 0.5% overhead for code pointers but ∼20% for all data pointers in nbench. We corroborate this effect using a pointer-chasing microbenchmark that measures the cost of accessing signed pointers on the Apple M1 processor. Although such overheads may be acceptable for PACstyle deployments, they motivate architectural optimizations to reduce dereference latency. Prior designs like C3 [58] discuss the optimizations like predictions, showing close to zero protection overhead. But C3 design is based on Intel architectures, limiting some optimization.

Code pointers are dereferenced in branch, jump, or return instructions. The branch predictor will still work as is. Branch target prediction like branch target buffer (BTB) and return address stack (RAS) usually uses the PC of the current branch instruction for prediction instead of the pointer itself. Pointer encryption does not touch the predictor design, and thus, the prediction rate of the branch target will not be affected. With pointer encryption, the resolution of the branch will take one more cycle, which adds to the execution latency when a branch misprediction happens and has negligible overhead on a correct branch prediction. With a decent branch prediction, the performance overhead will be small, as shown in designs like PAC. We provide further proof of the effects of BTB and RAS on overhead in section VI.

