# A. Tackling Efficiency with Linearly Scalable FPMA

**FPMA Definition.** FPMA begins with the IEEE normalized floating-point representation [1]:

<span id="page-2-1"></span>
$$x = (-1)^{S_x} \cdot 2^{E_x - B} \cdot (1 + M_x) \tag{1}$$

where  $S_x$  is the sign bit,  $E_x$  the exponent,  $M_x$  the mantissa, and B the exponent bias. Based on Mitchell's logarithmic approximation [31]:  $\log_2(1+M_x) \approx M_x$ , FPMA approximates a floating-point number x in the logarithmic domain as:

$$\log_2(|x|) = E_x - B + \log_2(1 + M_x) \approx E_x - B + M_x \quad (2)$$

Then, the floating-point multiplication  $r = x \cdot y$  in logarithmic domain is approximated as:

$$\log_2(|r|) = \log_2(|x \cdot y|) \approx (E_x + M_x) + (E_y + M_y) - 2B$$
 (3)

Since the product r can also be denoted as  $\log_2(|r|) \approx E_r + M_r - B$ , it enables the approximate multiplication:

$$R \approx X + Y - B \tag{4}$$

where 
$$X = E_x + M_x$$
,  $Y = E_y + M_y$ , and  $R = E_r + M_r$ .

**FPMA's Potential.** FPMA fundamentally departs from conventional (approximate) multiplier-centric designs by *eliminating multipliers*: it replaces floating-point multiplication with integer addition over the concatenated exponent–mantissa field. This makes precision scaling slice-composable, where the datapath expands by chaining adder slices, yielding nearlinear (O(n)) growth in area and critical delay (Figure. 3b), rather than the superlinear cost of multiplier-based datapaths. As a result, FPMA can span aggressive low-bit modes and higher-precision execution (e.g., 8/16-bit) with modest overhead, enabling a single GEMM engine to sustain high throughput across mixed-precision LLM workloads.

