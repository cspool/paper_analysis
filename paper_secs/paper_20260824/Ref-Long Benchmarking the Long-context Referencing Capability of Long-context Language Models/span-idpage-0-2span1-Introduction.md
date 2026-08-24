# <span id="page-0-2"></span>1 Introduction

Long-context language models (LCLMs) have demonstrated remarkable long-context capabilities in tasks such as multi-document question answering [\(Bai et al.,](#page-9-0) [2024c;](#page-9-0) [Wang et al.,](#page-10-0) [2024\)](#page-10-0) and summarization [\(Liu et al.,](#page-10-1) [2024;](#page-10-1) [Laban et al.,](#page-10-2) [2024\)](#page-10-2). Among long-context capabilities, long-context referencing, referring to LCLMs' ability to correctly attribute interested items to specific parts of extensive long documents, is crucial and has many real-world applications. [1](#page-0-0) For instance, legal practi-

<span id="page-0-1"></span>> **[图片提取文字 (无描述)]:**
> ...Anthony ...Durant and and Jeremy Green played Lin work together on a together for team... New York... Indexed documents (3) with different ...Kobe almost ...Durant was a NBA player sharpshooter, convinced Paul LeBron was a to join his names. playmaker... team... Tell me the indexes of all The answer Query sections referencing Durant. is {1, 4}.
![](_page_0_Figure_9.jpeg)

Figure 1: An example Ref-Long task. Given a longcontext input with indexed documents that include several NBA players names, an LCLM is asked to give the indexes of documents that reference "Durant" (marked as red). Names other than "Durant" are marked as blue.

tioners need to quickly identify the specific chapter within the relevant legal code when presented with a particular case or provision, and financial professionals need to swiftly determine which financial report contains the given data.

Although various benchmarks exist for evaluating the long-context capabilities of LCLMs, very few assess the dimension of long-context referencing. Existing long-context benchmarks can be broadly categorized into two types. On one hand, *general long-context benchmarks*, such as Long-Bench [\(Bai et al.,](#page-9-0) [2024c\)](#page-9-0), L-Eval [\(An et al.,](#page-9-1) [2024\)](#page-9-1), NOCHA [\(Karpinska et al.,](#page-9-2) [2024\)](#page-9-2), and a combination of them (HELMET) [\(Yen et al.,](#page-10-3) [2024\)](#page-10-3), are either synthesized by adding irrelevant texts into short-context NLP tasks, which results in unrealistic context distributions and biased evaluations, or constructed from scratch with human annotations, which requires substantial resources and complicated human efforts. On the other hand, as a spe-

<span id="page-0-0"></span><sup>\*</sup>Equal contribution.

<sup>1</sup>The term *referencing* differs from *retrieval* in that it requires LCLMs to not only retrieve keys from long context,

but also know the location (specific parts) where these keys appears in the long context.

cific and well-studied type of long-context benchmark, *retrieval-based benchmarks* such as Needlein-a-Haystack [\(Kamradt,](#page-9-3) [2023a\)](#page-9-3), Counting-stars [\(Song et al.,](#page-10-4) [2024\)](#page-10-4), and RULER [\(Hsieh et al.,](#page-9-4) [2024\)](#page-9-4), focus on matching and retrieving target texts but often overlook the nuanced relationships between the retrieved texts and their surrounding contexts. This makes these benchmarks overly simplistic and not comprehensive. Moreover, while existing benchmarks address some aspects of long-context understanding, they fail to effectively evaluate longcontext referencing, highlighting the urgent need for practical and robust benchmarks in this area.

To address the above issues, this work proposes a novel benchmark called Referencing Evaluation for Long-context Language Models (Ref-Long), which is specifically designed to assess the longcontext referencing capability of LCLMs. As illustrated in Figure [1,](#page-0-1) given several indexed long documents and a query that includes "Durant", LCLMs are required to not only identify "Durant" in the given documents, but also need to figure out the indexes of documents that reference "Durant" rather than other NBA players. This task setting has several advantages. First, it considers the relationship information between the specific key and its surrounding context, which forces LCLMs to genuinely understand long contexts instead of simply relying on shortcuts to retrieve the key. As a result, [§4](#page-2-0) and [§5](#page-6-0) show that Ref-Long presents certain level of difficulty that challenges even the most advanced LCLMs (e.g., GPT-4o [\(Hurst et al.,](#page-9-5) [2024\)](#page-9-5)). Second, Ref-Long tasks can be constructed costefficiently, as only the locations of specific keys are required. Furthermore, as shown in [§4.3,](#page-4-0) Ref-Long tasks remain manageable for human annotators, allowing their difficulty level to be estimated.

Following the task setting, we construct three subsets, ranging from synthetic to realistic scenarios, to form the Ref-Long benchmark and evaluate 13 LCLMs. Experimental results on these subsets reveal that all LCLMs struggle with Ref-Long tasks ([§4,](#page-2-0) [§5\)](#page-6-0), highlighting their lack of long-context referencing capability. Furthermore, we investigate the challenge faced by LCLMs from several perspectives. Motivated by findings in [\(Wu et al.,](#page-10-5) [2025;](#page-10-5) [Yu et al.,](#page-10-6) [2025\)](#page-10-6) that LLMs may struggle with tasks easily handled by humans, we first conduct a human evaluation in [§4.3](#page-4-0) to assess whether task difficulty contributes to the challenges in Ref-Long. Next, we examine if the issue comes from the format of task queries by applying alternative formats

to evaluate LCLMs ([§4.4\)](#page-4-1). Additionally, we explore whether fine-tuning can mitigate LCLMs' limitations in long-context referencing ([§4.5\)](#page-5-0). Finally, we perform error analysis on LCLMs' failed cases ([§6\)](#page-7-0). In summary, our contributions are threefold.

- 1. First, we introduce Ref-Long, a novel benchmark that solves long-context referencing and limitations in existing long-context benchmarks.
- 2. Second, we demonstrate Ref-Long is uniquely challenging for state-of-the-art LCLMs yet remains accessible for human annotators, underscoring its importance for advancing the field.
- 3. Finally, through comprehensive analyses, we identify several findings that could be used to facilitate LCLMs' long-context referencing and understanding capabilities.

