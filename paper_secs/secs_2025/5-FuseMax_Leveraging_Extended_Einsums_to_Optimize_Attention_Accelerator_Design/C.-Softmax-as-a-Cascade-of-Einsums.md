# C. Softmax as a Cascade of Einsums

We now apply the same precise notation to the softmax. A softmax [5] over a 1-tensor is traditionally expressed with the following equation:

$$A_m = \frac{e^{I_m}}{\sum_k e^{I_k}} \tag{25}$$

In the context of attention, this operation becomes two dimensional and can be expressed using the following cascade with input  $QK_{m,n}$ :

$$SN_{m,p} = e^{QK_{m,p}} \tag{26}$$

$$SD_p = SN_{m,p} (27)$$

$$A_{m,n} = SN_{m,n}/SD_n \tag{28}$$

For each point in the iteration space (m, p), we exponentiate  $QK_{m,p}$  to generate the softmax numerator  $(SN_{m,p})$  in Einsum 26), reduce  $SN_{m,p}$  with addition to produce the softmax denominator  $(SD_p)$  in Einsum 27), and finally, divide the numerator and denominator to produce the final result  $(A_{m,p})$ in Einsum 28).

1) Improving Numerical Stability: Because  $e^{QK_{m,p}}$  can easily become extremely large, the above formulation suffers from overflow. Therefore, practical implementations [2], [42] often prefer the numerically stable variant that replaces Einsum 26 with:

$$GM_p = QK_{m,p} :: \bigvee_{m} \max(\cup)$$

$$SN_{m,p} = e^{QK_{m,p} - GM_p}$$
(30)

$$SN_{m,p} = e^{QK_{m,p} - GM_p} (30)$$

and drop the  $\frac{1}{\sqrt{E}}$  term when computing  $QK_{m,p}$ .<sup>4</sup> To compute the global maximum<sup>5</sup>  $GM_p$ , we reduce  $QK_{m,p}$  with the operator max (instead of +). Notice that subtracting  $GM_p$  from  $QK_{m,n}$  in the exponent is equivalent to dividing by  $e^{GM_p}$ , and because the  $\frac{1}{e^{GM_p}}$  term appears in both the numerator  $(SN_{m,p})$ via Einsum 30) and denominator ( $SD_p$  via Einsum 27), the result  $(A_{m,n})$  stays the same. This construction improves numerical stability by bounding the values of the softmax numerator  $SN_{m,p}$  to the range (0,1].

