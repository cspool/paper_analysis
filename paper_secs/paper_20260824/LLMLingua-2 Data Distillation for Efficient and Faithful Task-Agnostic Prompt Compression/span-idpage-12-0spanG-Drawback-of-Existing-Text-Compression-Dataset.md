# <span id="page-12-0"></span>G Drawback of Existing Text Compression Dataset

Existing extractive compression datasets such as SentComp [\(Filippova and Altun,](#page-9-9) [2013\)](#page-9-9) and Debate-

Sum [\(Roush and Balaji,](#page-10-5) [2020\)](#page-10-5) are mainly created for summarization task. The compressed texts provided in their dataset are usually too concise, only maintaining the main idea of the original text and lacking detailed information. This information loss inevitably hinders the downstream tasks such as document-based QA, as illustrated in Fig. [13](#page-16-0) and Fig. [14](#page-16-1)

#### <span id="page-13-0"></span>Our GPT-4 Instruction for Compression:

#### System Prompt:

You are an excellent linguist and very good at compressing passages into short expressions by removing unimportant words, while retaining as much information as possible.

#### User Prompt:

Compress the given text to short expressions, and such that you (GPT-4) can reconstruct it as close as possible to the original. Unlike the usual text compression, I need you to comply with the 5 conditions below:

- 1. You can ONLY remove unimportant words.
- 2. Do not reorder the original words.
- 3. Do not change the original words.
- 4. Do not use abbreviations or emojis.
- 5. Do not add new words or symbols.

Compress the origin aggressively by removing words only. Compress the origin as short as you can, while retaining as much information as possible. If you understand, please compress the following text: {*text to compress*} The compressed text is:

Figure 9: The instruction we used in GPT-4 compression.

## <span id="page-13-3"></span>Instruction1:

Could you please rephrase the paragraph to make it short, and keep 5% tokens?

#### Instruction2:

Summarize the provided examples in a few sentences, maintaining all essential reasoning aspects.

### Instruction3:

Remove redundancy and express the text concisely in English, ensuring that all key information and reasoning processes are preserved.

## Instruction4:

Follow these steps to shorten the given text content: 1. First, calculate the amount of information contained in each sentence, and remove sentences with less information. 2. Next, further condense the text by removing stop words, unnecessary punctuation, and redundant expressions. Refine the content while ensuring that all key information is retained. Let's do it step by step.

Figure 10: Other instructions we evaluated, which are proposed in LLMLingua [\(Jiang et al.,](#page-9-5) [2023a\)](#page-9-5).

