# <span id="page-5-3"></span>D. Enhanced optimization via commutation

Previous works have observed that employing the commutativity between CX gates exposes more optimization opportunities for SWAP insertion [44]. However, the commutation pattern they exploit is limited to a pair of CX gates, where they either act on the same control qubit or target qubit. In our findings, the general 2Q gate commutativity can be captured through the canonical form:

<span id="page-5-4"></span>**Theorem 1** (Canonical gate commutation). Let  $Can(a, b, c)_{q_0,q_1}$  and  $Can(a', b', c')_{q_1,q_2}$  denote canonical gates acting on qubits  $(q_0, q_1)$  and  $(q_1, q_2)$  respectively, with an overlapping qubit  $q_1$ . They are commutative if and only if

$$b = b' = c = c' = 0, (2)$$

that is, when both consist solely of XX rotations.

Detailed proof is in Appendix D. Through this formalized commutativity determination, the ordinary CX commutation pattern can be captured without tracking the control and target qubit positions, as shown in Fig. 7(a). Moreover, Fig. 7(b) showcases additional commutation patterns that are captured in the canonical form but remain difficult to handle for CX-based compilers. These patterns are commonly observed in real-world circuits (e.g., arithmetic, QFT, chemistry simulation) and the transformation to commutative canonical gates can be readily obtained using TKET.

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

(a) Efficient SWAP absorption via canonical commutation relations.

![](_page_6_Figure_2.jpeg)

<span id="page-6-1"></span>(b) More commutation pattern examples captured by the canonical form.

Fig. 7. Canonical representation efficiently captures commutative relations in real-world quantum circuits. (a) The canonical commutation relation enhances SWAP absorption opportunities in a formal and efficient manner. Herein commutativity within CX chain can be identified without tracking control and target qubit positions. (b) Additional commutation patterns captured in the canonical form. The first pattern is intuitive in the standard CX basis, while the subsequent three highlight complex equivalences obscured in the CX basis but clearly exposed in canonical form (C denotes 1Q Clifford).

