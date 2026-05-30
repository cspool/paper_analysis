# <span id="page-34-0"></span>**D.4. Speech Output**

Rather than training a speech generation model from the ground up, we leverage state-of-the-art pre-trained text-to-speech (TTS) systems to produce speech in relevant scenarios, and adapt our approach using a speech codec when needed. Our evaluation focuses on English omni-modal-in and voice-out, using two complementary metrics: mean opinion score (MOS; higher indicates greater naturalness) and TTS word error rate (WER; lower indicates higher intelligibility), the latter measured through an external ASR system. As reported in Table [20,](#page-35-0) existing off-the-shelf models already yield high-quality, neutral speech suitable for assistant-style applications. Among the back ends tested, OmniVinci-Magpie achieves the best overall balance (MOS **4.63**, WER **2.7**%), followed closely by gpt-4o-mini-tts (MOS 4.59, WER 3.1%) and Qwen-omni (MOS 4.53, WER 3.2%). OmniVinci-StableCodec delivers a competitive WER (2.9%) but with slightly reduced naturalness (MOS 4.12), highlighting that intelligibility and perceived naturalness are not always aligned. In contrast, Bark underperforms on both measures (MOS 3.32, WER 8.2%), consistent with its more stochastic synthesis approach.

**Setup.** We evaluate prompt following on VoiceBench style/control splits and conversational control tasks. We compare three prompting strategies over interleaved audio–vision contexts: (i) *Transcript prompting* (ASR→text): [aud*,* vis] <sup>×</sup><sup>3</sup> + text-prompt, (ii) *Native audio prompting* (encoder features): [aud*,* vis] <sup>×</sup><sup>3</sup> + aud-prompt, (iii) *TTS-injected prompting* (render text to speech, then encode): [aud*,* vis] <sup>×</sup><sup>3</sup>+TTS(text-prompt). We also ablate prompt position: *prefix* [aud-prompt] + [aud*,* vis] ×3 , *mid* [aud*,* vis]*,* [aud-prompt]*,* [aud*,* vis] ×2 , and *suffix* [aud*,* vis] ×3 *,* [aud-prompt].

**Metrics.** We report (a) *Prompt Adherence Rate* (PAR; judged by paired preference and rubric scoring), (b) *slot accuracy* for constrained commands (names, numerals, entities), and (c) latency/efficiency (no additional ASR pass). For speech rendering quality, MOS/WER results are summarized in Table [20.](#page-35-0)

**Key Insight 4.** (1) *Native audio prompting* is the most robust to accents, background noise, and overlapped speech; it preserves prosodic cues (rate, emphasis) that pure transcripts discard, leading to higher PAR and slot accuracy in noisy and accented conditions. (2) *Transcript prompting* is competitive on clean speech but degrades when ASR struggles on named entities or code-switched fragments. (3) *TTS-injected prompting* reduces acoustic mismatch in far-field scenarios and is effective when a consistent house voice is desired, but it transfers less speaker/style information than using the raw prompt audio. (4) Prompt *suffix* placement—immediately before the model's response—consistently outperforms prefix and mid insertion, likely due to reduced long-range interference in the attention context.

Encoding the *audio* prompt directly (no external ASR) yields the best prompt following under realistic noise/accents while lowering latency and memory by avoiding an extra ASR pass. Suffix-position audio prompts provide the strongest control.

Beyond raw scores, we observe consistent performance across synthesis regimes. Agentic cascaded setups that decouple text planning from acoustic rendering tend to produce strong MOS and low WER in our pipeline, while auto-regressive models are competitive but show greater variance. Importantly, swapping the TTS back end does not alter OmniVinci 's language understanding or response planning; it only affects the surface realization of speech, simplifying deployment-time customization (*e.g.*, voice, rate).

For interactive agents, streaming synthesis and low perceived latency are crucial. Our chosen back ends support incremental generation, enabling prompt first-audio while the remainder of the utterance is

<span id="page-35-0"></span>Table 20 | English naturalness MOS (higher is better) and TTS word error rate (WER; lower is better). Best per column in **bold**.

| Setup                     | Regime           | MOS ↑ | WER (%) ↓ |
|---------------------------|------------------|-------|-----------|
| Qwen-Omni                 | auto-regressive  | 4.53  | 3.2       |
| GPT-4o-mini               | –                | 4.59  | 3.1       |
| OmniVinci-CozyVoice       | agentic cascaded | 4.54  | 3.0       |
| OmniVinci-Bark            | agentic cascaded | 3.32  | 8.2       |
| OmniVinci-StableCodec     | auto-regressive  | 4.12  | 2.9       |
| OmniVinci-Magpie (chosen) | agentic cascaded | 4.63  | 2.7       |

synthesized. In production, we prioritize (i) stability on numerals, abbreviations, and named entities, (ii) speaker consistency across turns, and (iii) graceful handling of punctuation and prosody cues from text.