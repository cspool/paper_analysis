# <span id="page-2-1"></span>2 Existing Benchmarks for Evaluating LCLMs

Table 1 reviews recent efforts in developing benchmarks for LCLMs and discuss their scope, which motivates the development of our new benchmark. See §6 for broader related work.

<span id="page-2-0"></span>

|                                              | Complex<br>Procedure | High<br>Dispersion | >8K<br>Input | Real-world<br>Tasks | >1K<br>Output | Deterministic<br>Solutions |
|----------------------------------------------|----------------------|--------------------|--------------|---------------------|---------------|----------------------------|
| NIAH (Kamradt, 2023)                         | Х                    | Х                  | <b>√</b>     | Х                   | Х             | <b>✓</b>                   |
| RULER (Hsieh et al., 2024)                   | X                    | X                  | ✓            | X                   | X             | ✓                          |
| ZeroScrolls (Shaham et al., 2023)            | X                    | <b>√</b> *         | <b>√</b> *   | <b>√</b> *          | X             | <b>√</b> *                 |
| ∞Bench (Zhang et al., 2024)                  | X                    | X                  | ✓            | <b>√</b> *          | X             | <b>√</b> *                 |
| SumHaystack (Laban et al., 2024)             | X                    | X                  | ✓            | ✓                   | X             | X                          |
| HELMET (Yen et al., 2025)                    | ×                    | <b>√</b> *         | ✓            | <b>√</b> *          | X             | <b>√</b> *                 |
| LongGenBench <sub>1</sub> (Liu et al., 2024) | Х                    | ✓                  | <b>√</b>     | Х                   | <b>√</b>      | <b>√</b>                   |
| LongGenBench <sub>2</sub> (Wu et al., 2024)  | X                    | X                  | X            | X                   | ✓             | ✓                          |
| LongWriter (Bai et al., 2024)                | ×                    | X                  | X            | ✓                   | ✓             | X                          |
| LONGPROC (Ours)                              | <b>✓</b>             | ✓                  | <b>√</b> *   | <b>√</b> *          | ✓             | <b>✓</b>                   |

Table 1: Recent representative benchmarks for evaluating LCLMs. To the best of our knowledge, no previous benchmark encompasses all of the listed qualities.  $\checkmark$ \*: the quality is featured by a subset of the tasks in a benchmark.

Benchmarks focusing on long inputs. The majority of existing benchmarks focus on long-context recall challenges (the first block in Table 1). The needle-in-the-haystack test (Kamradt, 2023, NIAH), one of the most widely used benchmarks, requires models to retrieve a target statement (needle) embedded within irrelevant text (haystack). Several subsequent benchmarks have extended this paradigm by incorporating multiple needles (Hsieh et al., 2024; Li et al., 2024a; Laban et al., 2024), introducing multi-step (but limited to a few steps of) reasoning (Levy et al., 2024; Li et al., 2024a), or employing more contextually relevant content (Liu et al., 2023; Karpinska et al., 2024; Wang et al., 2024b; Vodrahalli et al., 2024). While these extensions add complexity to the original NIAH framework, these dataset variants still exhibit relatively low dispersion (Goldman et al., 2024). While some benchmarks incorporate more realistic tasks (e.g., summarization and many-shot in-context learning) requiring more than a few snippets of relevant contexts (Shaham et al., 2023; Zhang et al., 2024; Yen et al., 2025), they only involve short output lengths (typically less than 100 words).

Benchmarks focusing on long outputs. Recent efforts have started exploring benchmarks requiring longer outputs (the second block in Table 1). Liu et al. (2024); Xu et al. (2024b) concatenate multiple problems to test LCLMs' ability to solve multiple tasks in a single pass. Wu et al. (2024) evaluate LLMs' capacity to generate repetitive information (e.g., year-long diary entries) under range-related or periodic constraints. However, these tasks concatenate independent problems without meaningful logical dependencies and segments in the required outputs are often disjointed. Bai et al. (2024) evaluate LCLMs through long-form content generation (e.g., creating a 30,000-word article on Roman Empire history), but such open-ended tasks make evaluation inherently subjective. Furthermore, none of these existing benchmarks adequately examine models' capabilities in multi-step reasoning and procedure following.

<span id="page-3-0"></span>

|                    |     | .5K Le |       |     | 2K Le |       |     | 3K Lev |       | Access                     | Danasia   | Exec   |
|--------------------|-----|--------|-------|-----|-------|-------|-----|--------|-------|----------------------------|-----------|--------|
|                    | IN  | # In   | # Out | IN  | # In  | # Out | IN  | # In   | # Out | Info                       | Reasoning | Searcn |
| HTML to TSV        | 100 | 12K    | 0.5K  | 189 | 23K   | 1.3K  | 120 | 38K    | 3.7K  | √(sequential)              | _         | _      |
| Pseudocode to Code | 100 | 0.4K   | 0.3K  | 100 | 0.9K  | 0.7K  | _   | _      | _     | √(sequential)              | _         | _      |
| Path Traversal     | 100 | 1.2K   | 0.5K  | 100 | 4.8K  | 2.0K  | 100 | 12K    | 5.8K  | $\sqrt{\text{(targeted)}}$ | _         | _      |
| ToM Tracking       | 100 | 2.0K   | 0.5K  | 100 | 2.5k  | 2.0K  | 100 | 4.1K   | 7.9K  | √(sequential)              | $\sqrt{}$ | _      |
| Countdown          | 100 | 5.6K   | 0.5K  | 100 | 5.6K  | 1.7K  | 100 | 5.6K   | 6.5K  | _                          |           |        |
| Travel Planning    | _   | _      | _     | 100 | 6.0K  | 1.2K  | 100 | 6.0K   | 5.3K  | $\sqrt{\text{(targeted)}}$ | $\sqrt{}$ |        |

Table 2: Summary of tasks in LONGPROC. On the left, we show general statistics, including number of instances (N), and the average number of input and output tokens (# In/Out; counted with Llama-3 tokenizer). On the right, we compare tasks across three aspects on the requirements for accessing information, deductive reasoning, and executing search.

### 3 LONGPROC Benchmark

Recall that LONGPROC includes six diverse tasks. We provide task examples in Figure 1 and summarize the characteristics of these tasks in Table 2. In this section, we begin by describing the shared feature of **procedural generation** that underlies all tasks in LONGPROC. We then introduce each task in detail (§3.1) and analyze its distinct characteristics to highlight the diverse challenges they present for LCLMs (§3.2). Lastly, we explain the reliable evaluation metrics for these tasks (§3.3).

**Procedural generation.** The six tasks in LONGPROC share a common feature: each task requires LCLMs to execute a procedure to generate the output. Let  $\Sigma$  denote the vocabulary. Given an input  $\mathbf{X} \in \Sigma^*$ , a procedure  $\pi$  generates a **gold** output  $\mathbf{Y}^* \in \Sigma^*$ . The gold output  $\mathbf{Y}^*$  is composed of a sequence of entries  $\mathbf{Y}^* = \{y_1^*, y_2^*, ..., y_n^*\}$ , where each  $y_i^*$  is a structured form (e.g., text following a specific template such as a TSV row). The procedure  $\pi$  is deterministic: at step i, exactly one correct entry  $y_i^*$  exists, determined by  $\pi$  based on the task input and all previous entries, i.e.,  $y_i^* = \pi(\mathbf{X}, y_1^*, y_2^*, ..., y_{i-1}^*)$ .

We evaluate an LCLM by prompting it to execute an instruction  $\mathbf{I} \in \Sigma^*$  (describing  $\pi$ ) over  $\mathbf{X}$ :  $\mathbf{Y} = \mathrm{LCLM}(\mathbf{I}, \mathbf{X})$ . To clearly specify a procedure  $\pi$ , the instructions contain both detailed step-by-step descriptions about the procedure (see Figure 2 for a concrete example) and few-shot examples. Since  $\pi$  is deterministic, we can reliably evaluate the correctness of model actual prediction  $\mathbf{Y}$  by comparing each predicted entry  $y_i \in \mathbf{Y}$  against its corresponding gold entry  $y_i^* \in \mathbf{Y}^*$  using **rule-based metrics**.

#### <span id="page-3-1"></span>3.1 Tasks

We now introduce each of the tasks in LONGPROC with a focus on essential task characteristics. We discuss more detailed construction process in Appendix B and include **concrete examples for all tasks in Appendix** F.

HTML to TSV. This task (see top left of Figure 1 as well as Example F.1) requires LCLMs to extract query-specified information from HTML documents and organize the information into a table format (TSV). For instance, given an IMDB search result page in HTML format (with HTML tags) and a query specifying "extract the following properties from the items listed on the webpage: (1) Title; (2) Year; (3) Genre; (4) Rating", LCLMs are required to extract the data and format them into a table. We source the websites from Arborist (Li et al., 2024c) and manually annotate the questions and the ground truth TSV for the websites.

For this task, each entry  $y_i$  corresponds to the step of processing the i-th item from the HTML document and format it into a TSV row. The primary challenge of this task is to robustly extract all relevant information from HTML and format them correctly.

**Pseudocode to Code.** This task, introduced in SPoC (Kulal et al., 2019), requires translating pseudocode into C++ code. See Example F.2 for a concrete example. The pseudocode is structured line-by-line, with each line corresponding directly to a line of C++ code, maintaining a one-to-one mapping between source and target.

Similarly to the HTML to TSV task, each entry *y<sup>i</sup>* represents the processing of the *i*-th line of pseudocode in the input, while the processing performs translation from pseudocode to C++ code as opposed to merely bookkeeping.

**Path Traversal.** This task (see Example [F.3](#page-27-0) for a concrete example) requires LCLMs to keep track of a route between two nodes, represented by a set of cities, in a graph where each city has **exactly one** outgoing connection to another city (node). Given a description of city connections and a source-destination pair, LCLMs must output a step-by-step route. It is important to note that this task does **not** require searching over a graph, since by construction, we constrain each city to have one and only one outgoing city, and we guarantee there exists one unique path from the source city to the destination city.

An entry *y<sup>i</sup>* in this task corresponds to the step of visiting to one city along the route, where each step transits into the unique outgoing connection from the current city to another city. This task challenges LCLMs to correctly retrieve the the connected outgoing city from the input descriptions and format the step into a standardized format at each step.

**Theory-of-Mind Tracking.** Inspired by a series of theory-of-mind reasoning datasets [\(Le](#page-12-6) [et al.,](#page-12-6) [2019;](#page-12-6) [Sclar et al.,](#page-13-6) [2023;](#page-13-6) [He et al.,](#page-11-8) [2023;](#page-11-8) [Sprague et al.,](#page-14-6) [2024\)](#page-14-6), we design this task which requires tracking a person' beliefs about object locations in a dynamic environment. Given a story involving a sequence of person and object placements, LCLMs are requested to determine a person's belief about a specific object's location while considering the person's limited perspective. See bottom left of Figure [1](#page-1-0) for an example.

This tasks differs from the previous three tasks by its requirement for **deductive reasoning**. Specifically, determining whether an person's belief should be updated requires inferring whether the person can observe the object (i.e., whether they are in the same room). The entry *y<sup>i</sup>* at a step *i* records the detailed reasoning process by tracking the person's location, the object's location, the visibility condition, and the resulting belief state.

**Countdown.** Countdown is a game requiring to reach a target number using a list of given numbers along with four arithmetic operations (+, −, ×, and /) (see Figure [2](#page-4-0) for a simplified example and Example [F.5](#page-30-0) for a concrete one). We source the problems from [Gandhi et al.](#page-11-9) [\(2024\)](#page-11-9).

For countdown, we instruct LCLMs to perform a *depth-first-search procedure*, which tests their capabilities in terms of carrying out an exhaustive search robustly. The entry *y<sup>i</sup>* at step *i* is a filled template recording the state (the current set of numbers) and the actions (choosing two numbers to apply an operation or backtrack to previous states) taken at the state.

**Travel Planning.** This task (see right of Figure [1](#page-1-0) for an example) adapted from [Zheng et al.](#page-15-2) [\(2024\)](#page-15-2) requires LCLMs to generate a multi-city travel plan that satisfies various constraints including fixed schedules, city visit durations, and direct flight availability between cities.

We also instruct the models to perform a depth-first-search procedure for this task.

