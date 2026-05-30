# L Limitation

DartQuant is tailored for uniformly distributed integer formats. Its effectiveness on alternative formats, such as FP4, or other non-uniform numerical representations, remains to be further explored and validated. Furthermore, Whip assumes that activations are approximately zero-mean, which generally holds true for most transformer layers. However, in rare cases where the activation mean significantly deviates from zero, the effectiveness of Whip may degrade.

In the future, we can explore a wider range of distribution transformation methods to better accommodate various numerical formats.

