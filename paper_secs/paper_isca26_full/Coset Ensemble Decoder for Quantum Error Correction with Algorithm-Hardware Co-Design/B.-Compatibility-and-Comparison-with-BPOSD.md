# B. Compatibility and Comparison with BP+OSD

To position our decoder on the accuracy spectrum between UF and near-optimal decoders, we include BP+OSD as an accuracy reference alongside MWPM and UF, using product-sum BP with OSD-CS of order 15 on the same Tanner graph for a fair comparison. The evaluation uses the repetition code

![](_page_11_Figure_10.jpeg)

<span id="page-11-4"></span>Fig. 18. Decoding performance of different random policies.

under a phenomenological noise model, which additionally verifies the generality of our decoder across code families and noise models. We sweep code distances  $d \in \{5,7\}$  and physical error rates  $p \in [0.04,0.08]$ . Fig. 16 reports the LER of MWPM, UF, BP+OSD, and our decoder, with the left and right panels corresponding to d=5 and d=7, respectively. Our decoder achieves LER within  $1.0-1.4\times$  of MWPM, on par with BP+OSD  $(1.0-1.7\times)$ , while UF trails by  $2.7-5.7\times$ . On these benchmarks, our decoder tracks MWPM and BP+OSD closely, decisively separating it from UF.

We also evaluate our decoder under biased phenomenological noise with three common bias ratios  $\eta=p_Z/p_X$ :  $\eta{=}0.5$  (X-biased),  $\eta{=}1$  (depolarizing), and  $\eta{=}10$ . As shown in Fig. 17, our decoder closes  $\sim\!94\%$  of the UF-to-MWPM gap under X-biased noise, where vanilla UF incurs  $6.2\times$  higher LER than MWPM. The corresponding system-infidelity curves are shown in the right panel of Fig. 17.

# B. Compatibility and Comparison with BP+OSD

To position our decoder on the accuracy spectrum between UF and near-optimal decoders, we include BP+OSD as an accuracy reference alongside MWPM and UF, using product-sum BP with OSD-CS of order 15 on the same Tanner graph for a fair comparison. The evaluation uses the repetition code

![](_page_11_Figure_10.jpeg)

<span id="page-11-4"></span>Fig. 18. Decoding performance of different random policies.

under a phenomenological noise model, which additionally verifies the generality of our decoder across code families and noise models. We sweep code distances  $d \in \{5,7\}$  and physical error rates  $p \in [0.04,0.08]$ . Fig. 16 reports the LER of MWPM, UF, BP+OSD, and our decoder, with the left and right panels corresponding to d=5 and d=7, respectively. Our decoder achieves LER within  $1.0-1.4\times$  of MWPM, on par with BP+OSD  $(1.0-1.7\times)$ , while UF trails by  $2.7-5.7\times$ . On these benchmarks, our decoder tracks MWPM and BP+OSD closely, decisively separating it from UF.

We also evaluate our decoder under biased phenomenological noise with three common bias ratios  $\eta=p_Z/p_X$ :  $\eta{=}0.5$  (X-biased),  $\eta{=}1$  (depolarizing), and  $\eta{=}10$ . As shown in Fig. 17, our decoder closes  $\sim\!94\%$  of the UF-to-MWPM gap under X-biased noise, where vanilla UF incurs  $6.2\times$  higher LER than MWPM. The corresponding system-infidelity curves are shown in the right panel of Fig. 17.

