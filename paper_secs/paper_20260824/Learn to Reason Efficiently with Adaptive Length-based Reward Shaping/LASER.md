# LASER

$$C(y) = R(x, y)$$

$$\lambda(y) = \mathbb{I}(R)$$

$$S(y) = \alpha \cdot \mathbb{I}(L(y) < L_T)$$

where  $L_T = 10$ .

### LASER-D

$$C(y) = R(x, y)$$

$$\lambda(y) = \mathbb{I}(R)$$

$$S(y) = \alpha \cdot \mathbb{I}(L(y) < L_A)$$

where  $L_A \in \{10, 7.5, 5\}$ .

### LASER-DE

$$\begin{split} C(y) &= R(x,y) \\ \lambda(y) &= 1 \\ S(y) &= \alpha \cdot \mathbb{I}(R) \cdot \mathbb{I}(L(y) \leq L_A) + \alpha \cdot (1 - \mathbb{I}(R)) \cdot \mathbb{I}(L(y) > L_A) \end{split}$$
 where  $L_A \in \{12.5, 10, 7.5\}.$ 

For methods with multiple adaptive target lengths L<sup>A</sup> values (ThinkPrune, LASER-D, and LASER-DE), different shades of the base colors were used:

- Correct responses (blue): RGB(26,71,142), RGB(62,101,184), RGB(125,154,230)
- Incorrect responses (red): RGB(139,0,0), RGB(183,50,40), RGB(224,93,86)

