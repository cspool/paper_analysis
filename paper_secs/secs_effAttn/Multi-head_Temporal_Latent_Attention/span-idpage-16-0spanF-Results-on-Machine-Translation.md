# <span id="page-16-0"></span>F Results on Machine Translation

<span id="page-16-1"></span>Table 9: Machine translation BLEU (↑) results on WMT14 [\[6\]](#page-10-16) English-German translation

| Model on WMT14 En-De | BLEU  |
|----------------------|-------|
| MLA                  | 25.63 |
| Proposed MTLA        | 25.57 |

While the focus of this paper is on long-sequence tasks, text-based translation generally involves much shorter sequences than speech translation. Nevertheless, this section presents MTLA results on the machine translation task using WMT14 English–German data. As shown in Table [9,](#page-16-1) MTLA achieves competitive performance compared to MLA, demonstrating that context compression does not degrade performance on this task.

## G Assets and licenses

The following licenses apply to the datasets used in this paper:

- CC-BY-NC-ND-4.0: <https://spdx.org/licenses/CC-BY-NC-ND-4.0> applies to MuST-C data.
- CC-BY-SA-4.0: <https://spdx.org/licenses/CC-BY-SA-4.0> applies to XSum data.
- CC BY 4.0: <https://spdx.org/licenses/CC-BY-4.0> applies to AMI data.
- CC BY-NC 4.0: <https://spdx.org/licenses/CC-BY-NC-4.0> applies to SLURP data.

The following license applies to the code and Python package used in this paper:

• Apache-2.0: applies to Fairseq ([https://github.com/facebookresearch/fairseq/](https://github.com/facebookresearch/fairseq/blob/main/LICENSE) [blob/main/LICENSE](https://github.com/facebookresearch/fairseq/blob/main/LICENSE)).