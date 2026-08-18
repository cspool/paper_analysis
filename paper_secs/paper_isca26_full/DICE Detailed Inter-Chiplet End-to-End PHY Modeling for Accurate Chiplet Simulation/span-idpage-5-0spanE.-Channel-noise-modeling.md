# <span id="page-5-0"></span>E. Channel noise modeling

After modulation symbols are transmitted over the interchiplet channel where *noise* (in the form of channel, jitter, and cross talk noise) is added.

**AWGN channel noise.** DICE models the channel noise as additive white Gaussian noise (AWGN), accounting for two dominant high-speed link impairments—1) *jitter* [32] and 2) *crosstalk* (XT) [33]. Together with the channel's intrinsic operating conditions, these impairments determine link quality, which we quantify using signal-noise ratio (SNR) [34], [35].

**Jitter.** Jitter refers to random variations in signal timing that translate clock-edge uncertainty into waveform errors (samples too early or late), thereby reducing SNR, closing the eye diagram, and degrading link quality. A widely used jitter model [36]–[38] is:

$$SNR_{jitter,linear} \approx \left(\frac{T_{sym}}{\pi \sigma_t}\right)^2 \implies SNR_{jitter,dB} \approx 20 \log_{10} \left(\frac{T_{sym}}{\pi \sigma_t}\right), \quad (4)$$

where  $\sigma_t$  denotes the root-mean-square (RMS) clock-edge timing error and  $T_{\rm sym}$  the symbol period. In DICE, we set  $T_{\rm sym}$  according to the network clock rate (32 Gb/s, following AMD's Infinity Fabric) and assume  $\sigma_t \approx 1\,\mathrm{ps}$  [39], [40], yielding an average SNR<sub>jitter,dB</sub>  $\approx 26.0\,\mathrm{dB}$ .

**Crosstalk (XT).** XT is hardware-induced coupling from aggressor lanes into a victim channel; its severity depends on interconnect geometry (*e.g.*, wire spacing and length) [41] and the modulation format [35], [42]. By default, DICE sets  $SNR_{XT,dB} \approx 20.0 \, dB$  guided by public data [43].

Calculating effective SNR and BER. We quantify the end-to-end link quality using an *effective* SNR that aggregates independent impairment sources. Let SNR<sub>base</sub> represent the intrinsic channel noise determined by its physical characteristics and operating conditions. Converting all SNR terms into *linear* scale, the effective SNR is obtained using the harmonic-sum rule [14]:

<span id="page-5-4"></span>
$$\frac{1}{SNR_{eff,linear}} = \frac{1}{SNR_{base,linear}} + \frac{1}{SNR_{jitter,linear}} + \frac{1}{SNR_{XT,linear}}.$$
 (5)

**Error injection.** Driven by SNR<sub>eff</sub>, DICE implements an *error injector* in gem5 (Listing 1) that corrupts transmitted PAM4 symbols with AWGN. As shown in Listing 1, with Gray mapping and  $X = \{-3d, -d, +d, +3d\}$  at  $d = 50 \,\text{mV}$  [9], [42], a symbol  $x \in X$  is transmitted as:

$$y = x + n, \qquad n \sim \mathcal{N}(0, \sigma^2),$$

where  $\sigma^2$  is the voltage noise variance implied by SNR<sub>eff</sub>. With average energy per symbol  $E_s = \frac{1}{4} \sum_{x \in \{-3d, -1d, 1d, 3d\}} x^2 = 5d^2$ ,  $\sigma$  is given by:

$$\sigma^2 = \frac{E_s}{\text{SNR}_{\text{eff.linear}}} = \frac{5d^2}{\text{SNR}_{\text{eff.linear}}}.$$

# <span id="page-5-0"></span>E. Channel noise modeling

After modulation symbols are transmitted over the interchiplet channel where *noise* (in the form of channel, jitter, and cross talk noise) is added.

**AWGN channel noise.** DICE models the channel noise as additive white Gaussian noise (AWGN), accounting for two dominant high-speed link impairments—1) *jitter* [32] and 2) *crosstalk* (XT) [33]. Together with the channel's intrinsic operating conditions, these impairments determine link quality, which we quantify using signal-noise ratio (SNR) [34], [35].

**Jitter.** Jitter refers to random variations in signal timing that translate clock-edge uncertainty into waveform errors (samples too early or late), thereby reducing SNR, closing the eye diagram, and degrading link quality. A widely used jitter model [36]–[38] is:

$$SNR_{jitter,linear} \approx \left(\frac{T_{sym}}{\pi \sigma_t}\right)^2 \implies SNR_{jitter,dB} \approx 20 \log_{10} \left(\frac{T_{sym}}{\pi \sigma_t}\right), \quad (4)$$

where  $\sigma_t$  denotes the root-mean-square (RMS) clock-edge timing error and  $T_{\rm sym}$  the symbol period. In DICE, we set  $T_{\rm sym}$  according to the network clock rate (32 Gb/s, following AMD's Infinity Fabric) and assume  $\sigma_t \approx 1\,\mathrm{ps}$  [39], [40], yielding an average SNR<sub>jitter,dB</sub>  $\approx 26.0\,\mathrm{dB}$ .

**Crosstalk (XT).** XT is hardware-induced coupling from aggressor lanes into a victim channel; its severity depends on interconnect geometry (*e.g.*, wire spacing and length) [41] and the modulation format [35], [42]. By default, DICE sets  $SNR_{XT,dB} \approx 20.0 \, dB$  guided by public data [43].

Calculating effective SNR and BER. We quantify the end-to-end link quality using an *effective* SNR that aggregates independent impairment sources. Let SNR<sub>base</sub> represent the intrinsic channel noise determined by its physical characteristics and operating conditions. Converting all SNR terms into *linear* scale, the effective SNR is obtained using the harmonic-sum rule [14]:

<span id="page-5-4"></span>
$$\frac{1}{SNR_{eff,linear}} = \frac{1}{SNR_{base,linear}} + \frac{1}{SNR_{jitter,linear}} + \frac{1}{SNR_{XT,linear}}.$$
 (5)

**Error injection.** Driven by SNR<sub>eff</sub>, DICE implements an *error injector* in gem5 (Listing 1) that corrupts transmitted PAM4 symbols with AWGN. As shown in Listing 1, with Gray mapping and  $X = \{-3d, -d, +d, +3d\}$  at  $d = 50 \,\text{mV}$  [9], [42], a symbol  $x \in X$  is transmitted as:

$$y = x + n, \qquad n \sim \mathcal{N}(0, \sigma^2),$$

where  $\sigma^2$  is the voltage noise variance implied by SNR<sub>eff</sub>. With average energy per symbol  $E_s = \frac{1}{4} \sum_{x \in \{-3d, -1d, 1d, 3d\}} x^2 = 5d^2$ ,  $\sigma$  is given by:

$$\sigma^2 = \frac{E_s}{\text{SNR}_{\text{eff.linear}}} = \frac{5d^2}{\text{SNR}_{\text{eff.linear}}}.$$

