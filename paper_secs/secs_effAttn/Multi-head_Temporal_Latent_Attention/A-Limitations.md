# A Limitations

Due to limited computational resources, this work does not investigate large language model (LLM) pre-training. The proposed MTLA is designed specifically for decoder-only architectures and can efficiently compress the KV cache. Standard text-based LLMs are successful examples of decoderonly models. Recent studies have shown that pre-pending speech representations as prompts before the self-attention input can extend text-based LLMs to speech tasks. However, building an LLM based on MTLA or replacing self-attention in a pre-trained LLM with MTLA and re-training it requires very substantial computational resources, which we do not possess. As a result, we are unable to construct an LLM based on MTLA to verify its extension to other tasks, such as speech. Instead, we construct decoder-only models and train them from scratch to evaluate MTLA across a range of tasks.

Second, as Transformer-based models have been extensively developed by the community in recent years, there is a large amount of related work. It is not feasible for us to implement and compare all such approaches. In this work, we compare MTLA with the most relevant and representative KV-cache compression methods, including MQA, GQA, and MLA. In addition, we also include comparisons with the typical token compression method SnapKV and the state space model Mamba-2. Further comparisons are only feasible through theoretical discussion, as presented in Section [2.](#page-1-0)

This work focuses on long-sequence tasks, particularly speech, due to the naturally long sequence length of speech inputs. We also conduct evaluations on a text summarisation task. While many additional tasks could be used to further evaluate its effectiveness of MTLA, we leave such investigations for future work. Given the growing dimensionality of modern LLMs and the increasing use of long reasoning chains to improve output quality, the MTLA, which compresses the KV cache along both the latent and temporal dimensions, can be particularly valuable.

## B Broader impact

Decoder-only architectures based on self-attention have become increasingly popular in recent years, especially in the context of large language models (LLMs). However, due to their high dimensionality and massive number of parameters, LLMs incur expensive inference costs and are heavily dependent on GPUs. This problem is further exacerbated by the use of chain-of-thought, which enhances reasoning ability but results in significantly longer output sequences, making inference even more costly. Such inference consumes substantial energy from GPUs. By contrast, our proposed MTLA compresses the Key-Value Cache in both latent and temporal dimensions, greatly improving inference efficiency, which can be of great value to make LLMs more energy-efficient and environmentally sustainable. Therefore, our work has the potential to generate a positive societal impact. We do not know of any negative societal impact.

## <span id="page-13-0"></span>C Data Set Statistics

The ST task uses the MuST-C [\[16\]](#page-10-10) v1.0 English-German (En-De) dataset, with data preprocessing following the Fairseq example, using 8,000 unigrams as the target language modelling units and fbank features as input. The text summarisation task is conducted on the XSum [\[31\]](#page-11-11) dataset, where 30,000 BPE units are used. For the ASR task, the AMI [\[8\]](#page-10-11) dataset is employed. Due to the challenging nature of the data, fixed WavLM [\[9\]](#page-10-15) Large features are extracted using the S3PRL [\[48\]](#page-12-14) toolkit as input. When measuring inference speed, this feature is pre-stored and 100 BPE units are used. For the SLU task, the SLURP [\[5\]](#page-10-12) dataset is used to evaluate intent classification, with fbank features as input. Following [\[3\]](#page-10-13), intent classification is performed by jointly predicting the transcription and the intent to achieve better performance. A total of 500 BPE units are used for transcription modelling.

The data set statistics for the datasets used in the experiments are shown in Table [7.](#page-14-1) The MuST-C [\[16\]](#page-10-10) v1.0 En-De dataset comprises English-German speech translation data collected from TED Talks. The Augmented Multi-Party Interaction (AMI) Meeting Corpus [\[8\]](#page-10-11) offers 100 hours of English meeting recordings captured in instrumented rooms, featuring multimodal data such as audio, video, and whiteboard content, with annotations including speech transcriptions and dialogue acts. The Spoken Language Understanding Resource Package (SLURP) [\[5\]](#page-10-12) dataset is a comprehensive English spoken language understanding resource encompassing 18 domains, designed to facilitate tasks like intent

classification and slot filling, with a diverse set of utterances. The XSum [31] dataset consists of BBC news articles from 2010 to 2017, each paired with a single-sentence abstractive summary, totalling over 226K document-summary pairs, and is widely used for evaluating summarisation models.

<span id="page-14-1"></span>Table 7: Statistics of datasets used in this paper

| Table 7. Statistics of datasets used in this paper |                    |                                   |  |  |  |  |
|----------------------------------------------------|--------------------|-----------------------------------|--|--|--|--|
|                                                    | MuST-C v1.0 En-De  |                                   |  |  |  |  |
| Domain                                             | TED Talk           |                                   |  |  |  |  |
| Train set                                          | train              |                                   |  |  |  |  |
| -Duration                                          |                    | 400.0 hours                       |  |  |  |  |
| -German words                                      |                    | 3880K                             |  |  |  |  |
| Test sets                                          | dev                | tst-COMMON                        |  |  |  |  |
| -Duration                                          | 2.3 hours          | 4.1 hours                         |  |  |  |  |
| -German words                                      | 26K                | 44K                               |  |  |  |  |
|                                                    |                    | XSum Dataset                      |  |  |  |  |
| Domain                                             |                    | BBC News Articles                 |  |  |  |  |
| Train set                                          |                    | train                             |  |  |  |  |
| -Documents                                         |                    | 204K                              |  |  |  |  |
| -Avg. article length                               | 431 words          |                                   |  |  |  |  |
| -Avg. summary length                               | 23 words           |                                   |  |  |  |  |
| Test sets                                          | dev test           |                                   |  |  |  |  |
| -Documents                                         | 11K 11K            |                                   |  |  |  |  |
|                                                    | AMI Meeting Corpus |                                   |  |  |  |  |
| Domain                                             | Meetings           |                                   |  |  |  |  |
| Train set                                          |                    | train                             |  |  |  |  |
| -Duration                                          |                    | 100.0 hours                       |  |  |  |  |
| -Utterances                                        |                    | 108K                              |  |  |  |  |
| Test sets                                          | dev                | test                              |  |  |  |  |
| -Utterances                                        | 13K                | 12K                               |  |  |  |  |
|                                                    |                    | SLURP Dataset                     |  |  |  |  |
| Domain                                             | Human-Con          | nputer Interaction (HCI) commands |  |  |  |  |
| Train set                                          | train              |                                   |  |  |  |  |
| -Duration                                          | 83.7 hours         |                                   |  |  |  |  |
| -Utterances                                        |                    | 120K                              |  |  |  |  |
| Test sets                                          | dev                | test                              |  |  |  |  |
| -Duration                                          | 6.9 hours          | 10.3 hours                        |  |  |  |  |
| -Utterances                                        | 9K 13K             |                                   |  |  |  |  |

