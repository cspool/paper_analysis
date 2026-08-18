# *B. At-Rest Memory Security*

To ensure data confidentiality, all writes to NVM must be encrypted. For this purpose, this paper leverages a tweakable block cipher mode (AES-XTS), widely adopted in the industry for memory (AMD SME [53] and Intel TDX [1]) and storage (e.g. Microsoft Bit-Locker [75]), among others. Figure 2 illustrates the operation of AES-XTS, where two AES keys are used together with a tweak (address). The upshot is that AES-XTS *ensures*

![](_page_2_Picture_6.jpeg)

Fig. 2: AES-XTS [26].

*data confidentiality without an integrity tree for verification*. In contrast, AES-CTR (counter-encryption mode) requires an integrity tree not only to protect data integrity, but also to protect confidentiality because it relies on counter freshness guarantee [83]. That is, AES-XTS is not susceptible to the counter replay attacks [83] that breaks the confidentiality guarantee—though it cannot prevent data tampering due to the lack of integrity verification. Also, AES-XTS provides stronger confidentiality guarantees than AES-CTR [87], e.g., AES-CTR is vulnerable to single-bit-flip attacks whereas AES-XTS remains resistant.

