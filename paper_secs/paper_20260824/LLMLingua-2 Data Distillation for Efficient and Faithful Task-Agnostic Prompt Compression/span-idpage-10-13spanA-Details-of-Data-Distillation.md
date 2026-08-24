# <span id="page-10-13"></span>A Details of Data Distillation

To construct the extractive compression dataset, we use GPT-4-32k to compress the original meeting transcript. Each transcript is divided into chunks first, with each chunk terminating at the end of a complete sentence and not exceeding 512 tokens. We employ the default parameter settings with a temperature of 0.3 and a top\_p of 1.0. The maximum number of generated tokens is set to 4096. Transcripts exceeding 28K tokens are truncated, allowing a 4K token budget for generation. Fig. [9](#page-13-0) presents the full instruction used in GPT-4 compression. Tab. [8](#page-10-15) shows the statistics of our MeetingBank compression dataset.

<span id="page-10-15"></span>

| Data Part  |       |        | Data Size Chunk Sentence (Avg) Token (Avg) 1/τ |       |       |
|------------|-------|--------|------------------------------------------------|-------|-------|
| Original   | 5,169 | 41,746 | 232                                            | 3,635 | -     |
| Compressed | 5,169 | 41,746 | 132                                            | 1,415 | 2.57x |

Table 8: Statistics of MeetingBank compression dataset.

## <span id="page-10-12"></span>B Details of Data Annotation

Based on the compressed prompt, we design a word annotation algorithm to automatically assign each word a label indicating whether the word in the original prompt should be retained. Initially, all labels of the original words are set to *False*. Then, for every word in the compressed prompt, we search for its corresponding word in the original prompt, which is then assigned a *True* label.

Sliding Window: To assign labels to the appropriate words in the original prompt, we utilize a sliding window approach, constraining the search scope within a local window centered on the previously matched word in the original prompt.

## <span id="page-11-1"></span>Prompt Compression Details:

#### Example 1:

Item 15, report from City Manager Recommendation to adopt three resolutions. First, to join the Victory Pace program. Second, to join the California first program. And number three, consenting to to inclusion of certain properties within the jurisdiction in the California Hero program. It was emotion, motion, a second and public comment. CNN. Please cast your vote. Oh. Was your public comment? Yeah. Please come forward. I thank you, Mr. Mayor. Thank you. Members of the council. My name is Alex Mitchell. I represent the hero program. Just wanted to let you know that the hero program. Has been in California for the last three and a half years. We're in. Over 20. We're in 28 counties, and we've completed over 29,000 energy efficient projects to make homes. Greener and more energy efficient. And this includes anything. From solar to water. Efficiency. We've done. Almost. \$550 million in home improvements.

## Example 2:

John: So, um, I've been thinking about the project, you know, and I believe we need to, uh, make some changes. I mean, we want the project to succeed, right? So, like, I think we should consider maybe revising the timeline.

Sarah: I totally agree, John. I mean, we have to be realistic, you know. The timeline is, like, too tight. You know what I mean? We should definitely extend it .

Figure 6: *LLMLingua-2* performs context awareness compression. The dark red highlights the words which are preserved at a 5x compression ratio, medium red denotes 3x compression ratio, and light red represents 2x compression ratio. Gray indicates discarded words during compression.

The search initiates from the last matching position. The *True* label is then assigned to the first matched word in the original prompt. Furthermore, the search is bidirectional to prevent mismatches caused by GPT-4's reordering, as shown in Fig. [5.](#page-4-0) Moreover, if GPT-4 introduces new words during compression, the sliding window restricts the search scope, preventing mismatches between the newly added words in the compressed prompt and words in the original prompt.

Fuzzy Matching: Another challenge arises from the "variation" misbehavior of GPT-4, as illustrated in Fig [5.](#page-4-0) GPT-4 may alter the original words in tense, voice, and singular/plural forms during compression, even when we request GPT-4 to compress by discarding words only. To address this issue, we first apply lemmatization to reduce words to their base form using Spacy[5](#page-11-3) , and then perform word matching using the sliding window approach.

## C Context Aware Compression

Fig. [6](#page-11-1) presents some compression results of our *LLMLingua-2* under different compression ratios. Our method effectively maintains the most meaningful words as the compression ratio increases.

## D Comparison with Baselines

In Fig. [11](#page-14-0) and Fig. [12,](#page-15-0) we qualitatively compare the compressed prompts of our methods with those of baseline method on GSM8K and BBH datasets. Note our *LLMLingua-2* here is only trained on

MeetingBank, but also yields more reasonable compressed prompt than baseline methods on the transferred domain data.

