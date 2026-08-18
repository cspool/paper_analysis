# Listing 1: Error injector pseudocode

```
1 /*Initialize random generator with SNR_eff,
2 which is determined by jitter, XT, SNR_base*/
3 sigma <- Es / SNR_eff_linear
4 rng <- NormalDist(mean=0, stddev=sigma)
5 for each flit in network:
6 msg_byte <- flit.get_msg_byte()
7 for each 2-bit PAM4 symbol in msg_byte:
8
```

<span id="page-5-2"></span>![](_page_5_Figure_16.jpeg)

Fig. 8: Error distribution under two example base SNRs. Takeaway: Higher SNR incurs fewer errors.

<span id="page-5-3"></span>![](_page_5_Figure_18.jpeg)

Fig. 9: Pre- and post-FEC FER sensitivity across varying base SNR values. Takeaway: when SNR falls below 25 dB (*i.e.*, in a highly noisy channel), 2-byte parity FEC becomes unreliable, necessitating either additional parity bytes or higher-level error detection mechanisms such as CRC. This paper adopts 35 dB as the baseline SNR based on publicly available data [10].

Following UCIe practice for short-reach inter-chiplet links, we assume a nominal SNR<sub>base,dB</sub>  $\approx 35.0\,\mathrm{dB}$  [10], [44]. After incorporating jitter and crosstalk impairments, this corresponds to an effective link quality of SNR<sub>eff,dB</sub>  $\approx 19.0\,\mathrm{dB}$ , which yields a baseline noise variance of  $\sigma \approx 12.7\,\mathrm{mV}$ . Lastly, beyond the defaults, DICE exposes all knobs for jitter level, crosstalk strength, and baseline SNR at run time to enable flexible DSE.

**Example: corrupting transmitted symbols.** Consider the PAM4 sequence x = [-50, -150, +150, +150] mV with a noise deviation of  $\sigma \approx 12.7 \text{ mV}$ . For a sample noise realization n = [+5.0, -21.0, -13.0, +8.0] mV, the transmitted waveform becomes y = x + n = [-45.0, -171.0, +137.0, +158.0] mV.

In Figure 8a and Figure 8b, we visualize PAM4 symbol distributions under AWGN for SNR<sub>base,dB</sub>=15.0 and 35.0, respectively. An *error symbol* is counted when noise drives a symbol across an adjacent voltage region (*e.g.*, -150 mV—

50 mV). We further report the pre-FEC FERs across a span of SNR<sub>base,dB</sub> in Figure 9.

As shown in Figure 8a and Figure 8b, increasing SNR<sub>base,dB</sub>—reflecting a cleaner, better-manufactured channel—reduces the number of error symbols. Further, with 2 parity bytes, Figure 9 shows that when the channel SNR falls below 25.0 dB, a growing fraction of errors become uncorrectable by FEC and must instead be handled at a higher protocol level, such as retransmission via CRC [45].

# Listing 1: Error injector pseudocode

```
1 /*Initialize random generator with SNR_eff,
2 which is determined by jitter, XT, SNR_base*/
3 sigma <- Es / SNR_eff_linear
4 rng <- NormalDist(mean=0, stddev=sigma)
5 for each flit in network:
6 msg_byte <- flit.get_msg_byte()
7 for each 2-bit PAM4 symbol in msg_byte:
8
```

<span id="page-5-2"></span>![](_page_5_Figure_16.jpeg)

Fig. 8: Error distribution under two example base SNRs. Takeaway: Higher SNR incurs fewer errors.

<span id="page-5-3"></span>![](_page_5_Figure_18.jpeg)

Fig. 9: Pre- and post-FEC FER sensitivity across varying base SNR values. Takeaway: when SNR falls below 25 dB (*i.e.*, in a highly noisy channel), 2-byte parity FEC becomes unreliable, necessitating either additional parity bytes or higher-level error detection mechanisms such as CRC. This paper adopts 35 dB as the baseline SNR based on publicly available data [10].

Following UCIe practice for short-reach inter-chiplet links, we assume a nominal SNR<sub>base,dB</sub>  $\approx 35.0\,\mathrm{dB}$  [10], [44]. After incorporating jitter and crosstalk impairments, this corresponds to an effective link quality of SNR<sub>eff,dB</sub>  $\approx 19.0\,\mathrm{dB}$ , which yields a baseline noise variance of  $\sigma \approx 12.7\,\mathrm{mV}$ . Lastly, beyond the defaults, DICE exposes all knobs for jitter level, crosstalk strength, and baseline SNR at run time to enable flexible DSE.

**Example: corrupting transmitted symbols.** Consider the PAM4 sequence x = [-50, -150, +150, +150] mV with a noise deviation of  $\sigma \approx 12.7 \text{ mV}$ . For a sample noise realization n = [+5.0, -21.0, -13.0, +8.0] mV, the transmitted waveform becomes y = x + n = [-45.0, -171.0, +137.0, +158.0] mV.

In Figure 8a and Figure 8b, we visualize PAM4 symbol distributions under AWGN for SNR<sub>base,dB</sub>=15.0 and 35.0, respectively. An *error symbol* is counted when noise drives a symbol across an adjacent voltage region (*e.g.*, -150 mV—

50 mV). We further report the pre-FEC FERs across a span of SNR<sub>base,dB</sub> in Figure 9.

As shown in Figure 8a and Figure 8b, increasing SNR<sub>base,dB</sub>—reflecting a cleaner, better-manufactured channel—reduces the number of error symbols. Further, with 2 parity bytes, Figure 9 shows that when the channel SNR falls below 25.0 dB, a growing fraction of errors become uncorrectable by FEC and must instead be handled at a higher protocol level, such as retransmission via CRC [45].

