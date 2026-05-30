# <span id="page-32-1"></span>**E Computing Costs**

Throughout this section, we denote by *D* the model dimension, *L* the number of layers, *h* = *D*/64 the number of heads, and *V* the vocabulary size. All models are trained with *V* = 65536.

The number of parameters for all Eagle models is computed by the formula:

$$\#(Params)_E = 13D^2L + 14DL + 4D + 2DV$$
 (23)

The FLOPs for inference is one forward pass for each token. It is approximated by twice the number of parameters (for matrices, there is one addition and one multiplication for each entry) plus six times the size of *WKV* internal states (see [7](#page-6-4) [8](#page-6-5) [9\)](#page-6-6), which is

$$#(InferFLOPs)_E = 2(13D^2L + 14DL + 4D + 2DV) + 6D^2L/h$$
(24)

$$= 26D^{2}L + 28DL + 8D + 4DV + 6D^{2}L/h$$
 (25)

The FLOPs for training are approximated as three times the FLOPs of the forward pass without the last term, yielding a total FLOPs of

$$#(TrainFLOPs)_E = 78D^2L + 84DL + 16D + 12DV + 18D^2L/h$$
(26)

These numbers for Finch are marginally larger:

$$\#(Params)_F = 13D^2L + 464DL + 4D + 2DV$$
 (27)

$$#(InferFLOPs)_F = 26D^2L + 928DL + 8D + 4DV + 6D^2L/h$$
(28)

$$\#(\text{TrainFLOPs})_{\text{F}} = 78D^2L + 2784DL + 24D + 12DV + 18D^2L/h \tag{29}$$

In both Eagle and Finch, one needs an internal state to store some previous information, just like any other RNN. In each layer, the internal state consists of three parts:

1. The most recent single-timestep input to the Time-mixing module, denoted as *xt*−<sup>1</sup> ∈ R *D*, useful for the Token Shift.

<span id="page-33-0"></span>

| Dataset                  | Domain             | Dataset                 | Domain              |
|--------------------------|--------------------|-------------------------|---------------------|
| Wikipedia <sup>a</sup>   | Encyclopedia       | marianna13/vault_text   | Books               |
| SlimPajama               | Web                | marianna13/random_quora | Forums              |
| peS2o                    | Academia           | marianna13/zlib         | Books               |
| BigPatent                | Patents            | minipile                | Various             |
| Pile of Law              | Legal, Administra- | tatoeba                 | Multilingual Trans- |
|                          | tive               |                         | lations             |
| StarCoder <sup>b</sup>   | Code               | poetry-foundation       | Poetry              |
| OSCAR23.01 <sup>c</sup>  | Multilingual Web   | proof-pile              | Academia: Math      |
| TED2020                  | Transcripts: TED,  | reddit-math             | Forums: Math        |
|                          | TEDx               | soda                    | Dialogue            |
| PhilPapers               | Academia: Philoso- | song_lyrics             | Lyrics              |
| •                        | phy                | TinyStories             | Stories             |
| NIH-ExPORTER             | Grants: NIH        | walkthroughs2020        | Game Walk-          |
| EuroParl                 | Multilingual Legal |                         | throughs            |
| Enron-Emails             | Emails             | wikihow-qa-16k          | How-To              |
| Ubuntu IRC               | Chat               | Alpaca                  | Various             |
| HackerNews               | Forums             | camel-ai/math           | Math                |
| OpenWebText2             | Web                | camel-ai/code           | Code                |
| Gutenberg PG-19          | Books              | camel-ai/physics        | Physics             |
| Books3                   | Books              | camel-ai/chemistry      | Chemistry           |
| OpenSubtitles            | Subtitles          | camel-ai/ai_society     | Job Roles           |
| YTSubtitles              | Subtitles          | camel-ai/biology        | Biology             |
| ao3_skylion              | Stories            | Dolly                   | Various             |
| honeyfeed-3600           | Stories            | Evol-Instruct           | Various             |
| scribble-17k             | Stories            | gpt4all                 | Code                |
| syosetu711k              | Stories (Japanese) | Guanaco                 | Various Multilin-   |
| marianna13/fanfics       | Stories            |                         | gual                |
| marianna13/gamedev       | Forums             | LaMini                  | Various             |
| marianna13/ia-books      | Books              | oasst1                  | Multilingual Con-   |
| marianna13/libgen        | Textbooks, Books   |                         | versations          |
| marianna13/research_gate | Academia           | ShareGPT                | Conversations       |
| marianna13/superuser     | Forums             | UltraChat               | Conversations       |
| marianna13/the-eye       | Books              | BELLE 10M Chinese       | Various Chinese     |

Table 9: Components of the RWKV World v2 dataset, their source links, and their domains.  $^a$ For Wikipedia, we include all languages from date 04/01/2023, with certain overrepresented languages randomly subsampled (see wiki.txt in the supplementary material for exact amounts)

 $<sup>^</sup>c$ For OSCAR23.01, we include non-English languages only, with certain languages randomly subsampled (see oscar.txt in the supplementary material for exact amounts)

| SlimPajama           | Soboleva et al. (2023)    |
|----------------------|---------------------------|
| StarCoder            | Li et al. (2023b)         |
| OSCAR23.01           | Suárez et al. (2019)      |
| TED2020              | Reimers & Gurevych (2020) |
| the Pile             | Gao et al. (2020)         |
| <b>Evol-Instruct</b> | Xu et al. (2023)          |

Table 10: RWKV World v2 dataset component citations

- 2. The most recent single-timestep input to the Channel-mixing module, denoted as  $x'_{t-1} \in \mathbb{R}^D$ , also useful in Token Shift.
- 3. WKV head memory: Denoted by  $wkv_{t,j} \in \mathbb{R}^{(D/h)\times (D/h)}$ , for  $j=1,2,\cdots,h$ . This is the core part of the internal state that dominates the most information.

The total size of the Eagle and Finch internal state is

$$\#(\text{State}) = L(2D + D^2/h) = 66DL$$
 (30)

<sup>&</sup>lt;sup>b</sup>For StarCoder, we included only those datasets with at least 10 stars

| Model Name | L  | D    | State Size | Parameters | InferFLOPs | TrainFLOPs |
|------------|----|------|------------|------------|------------|------------|
| Eagle 0.4B | 24 | 1024 | 1622016    | 4.62×108   | 9.33×108   | 2.80×109   |
| Eagle 1.5B | 24 | 2048 | 3244032    | 1.58×109   | 3.17×109   | 9.52×109   |
| Eagle 3B   | 32 | 2560 | 5406720    | 3.06×109   | 6.16×109   | 1.85×1010  |
| Eagle 7B   | 32 | 4096 | 8650752    | 7.52×109   | 1.51×1010  | 4.53×1010  |
| Finch 1.6B | 24 | 2048 | 3244032    | 1.60×109   | 3.22×109   | 9.66×109   |
| Finch 3B   | 32 | 2560 | 5406720    | 3.10×109   | 6.23×109   | 1.87×1010  |

Table 11: Released Eagle and Finch model details and FLOP counts. Inference and training FLOPs are per token numbers.

It's worth noting that the internal state size of Eagle and Finch is more than an order of magnitude bigger than RWKV-4 (which is 5*DL*). Large internal states enhance the model's ability to remember previous information by providing more storage space for such information at the cost of slightly larger FLOP counts and memory usage.

