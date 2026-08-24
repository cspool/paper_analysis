# N Preservation Priority in GPT-4 Compression

To gain insight into GPT-4's compression patterns, we analyze the distribution of different POS categories. Fig [16](#page-17-2) suggests that GPT-4 prioritizes the preservation of nouns, adjectives, and numerals, which typically play a more important role in the comprehension of the overall context.

## O Comparison With GPT-4 Compression

Table [9](#page-15-1) shows the comparison between LLMLingua-2 compressed prompts and GPT-4 compressed prompts. For GPT-4 compression, We use the same compression instruction as the one used in training data collection. The same chunking technique is also adopted with the chunk size setting to 512. It is shown that LLMLingua-2 achieves higher performance than GPT-4 compression on MeetingBank QA. We conjecture that LLMLingua-2's ability to learn compression knowledge from the entire dataset helps mitigate the influence of noise and information loss present

## <span id="page-15-0"></span>Original Prompt (249 tokens):

Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for \$10 each box. He rearranged five of these boxes into packages of six highlighters each and sold them for \$3 per package. He sold the rest of the highlighters separately at the rate of three pens for \$2. How much profit did he make in total, in dollars?

Let's think step by step

Sam bought 12 boxes x \$10 = \$120 worth of highlighters.

He bought 12 \* 30 = 360 highlighters in total.

Sam then took 5 boxes × 6 highlighters/box = 30 highlighters.

He sold these boxes for 5 \* \$3 = \$15

After selling these 5 boxes there were 360 - 30 = 330 highlighters remaining.

These form 330 / 3 = 110 groups of three pens.

He sold each of these groups for \$2 each, so made 110 \* 2 = \$220 from them.

In total, then, he earned \$220 + \$15 = \$235.

Since his original cost was \$120, he earned \$235 - \$120 = \$115 in profit.

The answer is 115

### Compressed prompt (144 tokens) by *LLMLingua*:

: Sam bought a dozen boxes each 30 highl pens inside, \$10 each. He reanged five of boxes into of

six each \$3 per. He sold the thelters separately at the of three \$2. much make total,

Lets think step

bought boxes x0 oflters

He 2 3ters in

Sam then boxes 6lters/box 0ters

He sold these boxes 5

Afterelling these boxes there 36030lters

ese00 of three

sold groups2 each so made \*2 \$20 from

In total, he015

Since his he \$ - \$120 = \$115 in profit.

The answer is 115

#### Compressed prompt (138 tokens) by *LLMLingua-2*:

Sam bought dozen 30 highlighter pens \$10 rearranged five boxes into six highlighters sold \$3 per sold rest three pens profit ? Sam bought 12 boxes x \$10 = \$120

12 \* 30 = 360 highlighters

5 boxes × 6 highlighters/box = 30

sold 5 \* \$3 = \$15

5 360 - 30 = 330 highlighters

330 / 3 = 110 groups three

sold \$2 110 \* 2 = \$220

earned \$220 + \$15 = \$235. original cost earned \$235 - \$120 = \$115

The answer is 115

Figure 12: Comparison with baseline. *LLMLingua-2* here is only trained on MeetingBank, but also yields more reasonable compressed prompt than *LLMLingua* on GSM8K.

<span id="page-15-1"></span>

| Methods           | QA    | Length |      |
|-------------------|-------|--------|------|
|                   | EM    | Tokens | 1/τ  |
| GPT-4 Compression | 84.86 | 1,221  | 2.5x |
| LLMLingua-2-small | 85.82 | 984    | 3.0x |
| LLMLingua-2       | 86.92 | 970    | 3.1x |
| Original          | 87.75 | 3,003  | 1.0x |

Table 9: Comparison with GPT-4 compressed prompt on MeetingBank.

