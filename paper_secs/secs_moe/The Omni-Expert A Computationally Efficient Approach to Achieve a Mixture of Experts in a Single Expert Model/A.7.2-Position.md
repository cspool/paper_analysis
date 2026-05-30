# A.7.2 Position

![](_page_36_Figure_1.jpeg)

Figure A7.2.1: Boxplots of (a) SRMR-CI and (b) STOI scores evaluated for three test datasets in all four room conditions using ratio masks for an LSTM network of 1 layer. Results are shown for the Omni-Expert model with predicted phonemes.

Table A7.2: Performance across different feature transformation locations, Estimated Marginal Mean (± 95% Confidence interval). Bold indicates the highest performance among the feature transformation locations.

|                              |                                                    | SRMR-CI                                                     |                                                             |                                                             |                                                             |
|------------------------------|----------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
|                              | InsertionPoint                                     | Church                                                      | Office                                                      | Lecture                                                     | Stairway                                                    |
| Phoneme<br>-Predicted<br>-OE | Hidden<br>Layer (H)<br>Input<br>Layer (I)<br>I + H | 1.364<br>(±0.010)<br>1.370<br>(±0.010)<br>1.385<br>(±0.010) | 1.988<br>(±0.014)<br>2.029<br>(±0.015)<br>2.039<br>(±0.015) | 1.750<br>(±0.013)<br>1.787<br>(±0.013)<br>1.801<br>(±0.013) | 1.952<br>(±0.015)<br>1.990<br>(±0.016)<br>1.996<br>(±0.015) |
| Phoneme<br>-Known            | H                                                  | 1.544<br>(±0.013)<br>1.616                                  | 2.030<br>(±0.015)<br>2.077                                  | 1.849<br>(±0.014)<br>1.956                                  | 2.028<br>(±0.016)<br>2.105                                  |
| -OE                          | I<br>I + H                                         | (±0.013)<br>1.643<br>(±0.013)                               | (±0.015)<br>2.076<br>(±0.015)                               | (±0.015)<br>1.960<br>(±0.015)                               | (±0.017)<br>2.109<br>(±0.017)                               |

|                   |                 | STOI              |                   |                   |                   |
|-------------------|-----------------|-------------------|-------------------|-------------------|-------------------|
|                   | Insertion Point | Church            | Office            | Lecture           | Stairway          |
| Phoneme           | H               | 0.778<br>(±0.002) | 0.818<br>(±0.002) | 0.789<br>(±0.002) | 0.829<br>(±0.002) |
| -Predicted<br>-OE | I               | 0.780<br>(±0.002) | 0.823<br>(±0.002) | 0.795<br>(±0.002) | 0.831<br>(±0.002) |
|                   | I + H           | 0.778<br>(±0.002) | 0.818<br>(±0.002) | 0.793<br>(±0.002) | 0.833<br>(±0.002) |
| Phoneme           | H               | 0.808<br>(±0.002) | 0.833<br>(±0.002) | 0.810<br>(±0.002) | 0.845<br>(±0.002) |
| -Known<br>-OE     | I               | 0.819<br>(±0.002) | 0.845<br>(±0.002) | 0.827<br>(±0.002) | 0.855<br>(±0.001) |
|                   | I + H           | 0.819<br>(±0.002) | 0.841<br>(±0.002) | 0.825<br>(±0.002) | 0.857<br>(±0.001) |

