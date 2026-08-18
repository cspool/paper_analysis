# Remapping the Redundant Negative Zero (Z-value).

Third, standard low-bit FP formats waste one code for "negative zero" [7]. DynFP reclaims this code and maps it to a useful normal value from the internal E3M2 range in Figure 12, either a finer-resolution intermediate value, or a larger outlier to extend dynamic range. If Z exceeds the group's format max  $F_{max}$ , the group becomes asymmetric, and its sign can be absorbed into the scaling factor with no extra datapath cost.

Together, these mechanisms define the numerical semantics of DynFP as follows:

$$v = \begin{cases} Z, & \text{if } E = 0, \ M = 0, \ S = 1, \\ (-1)^S \cdot 2^{1-B} \cdot M, & \text{if } E = 0, \ M \neq 0, \\ (-1)^S \cdot 2^{\Phi(I,\ell,E) - B} \cdot (1 + M), & \text{if } E \neq 0. \end{cases}$$

where S, E, M, and B denote the sign, exponent, mantissa, and bias, respectively. The parameter Z remaps the redundant negative-zero code, while the I-flag affects the effective exponent only through  $\Phi(I,\ell,E)$ . Collectively, these mechanisms allow a low-bit format to realize a small set of distribution-aware numerical behaviors in a unified format.

Hardware support. Figure 13 illustrates the datapath of the Unified Format Converter used to implement DynFP in hardware. The converter receives DynFP3/4 inputs and a per-group format index that selects the corresponding DynFP from the supported format set. The format parameters are realized through a small look-up table (LUT). The LUT uses the combined format index and data value as its lookup index, mapping DynFP encodings (e.g., with different E/M allocations or the I-flag enabled for gap insertion) to their equivalent E3M2 representations used for computation. To support the Z-value mechanism, the converter detects negative zero (-0) and selects a predefined Z-value in E3M2 format via a multiplexer; otherwise the LUT-converted value is used.

