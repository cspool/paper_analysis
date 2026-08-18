# VII. DISCUSSION

Broader Applicability of P<sup>3</sup> -LLM. Apart from HBM-PIM, the proposed low-precision PCU of P<sup>3</sup> -LLM is generally applicable to other DRAM variants such as GDDR [\[34\]](#page-13-31) and LPDDR [\[35\]](#page-13-32). For instance, the LPDDR-PIM [\[41\]](#page-14-12) introduced by Samsung adopts a similar PCU design as HBM-PIM, both containing a 16-way SIMD FP16 MAC unit that receives 256 bit inputs and weights to compute GEMV. Furthermore, the throughput-enhanced PCU in P<sup>3</sup> -LLM can also be applied to LPDDR, whose tCCD <sup>S</sup> is a half of tCCD <sup>L</sup>.

