# **Contents**

| 1 | Introduction |                                         | 3                                   |    |  |  |  |  |  |  |  |  |  |
|---|--------------|-----------------------------------------|-------------------------------------|----|--|--|--|--|--|--|--|--|--|
| 2 |              | Related Works                           |                                     |    |  |  |  |  |  |  |  |  |  |
|   | 2.1          |                                         | Typical Vision Encoders in VLMs<br> | 4  |  |  |  |  |  |  |  |  |  |
|   | 2.2          |                                         | End-to-end OCR Models               | 4  |  |  |  |  |  |  |  |  |  |
| 3 |              | Methodology                             |                                     |    |  |  |  |  |  |  |  |  |  |
|   | 3.1          |                                         | Architecture<br>                    | 5  |  |  |  |  |  |  |  |  |  |
|   | 3.2          |                                         | DeepEncoder<br>                     | 5  |  |  |  |  |  |  |  |  |  |
|   |              | 3.2.1                                   | Architecture of DeepEncoder<br>     | 5  |  |  |  |  |  |  |  |  |  |
|   |              | 3.2.2                                   | Multiple resolution support         | 6  |  |  |  |  |  |  |  |  |  |
|   | 3.3          |                                         | The MoE Decoder                     | 7  |  |  |  |  |  |  |  |  |  |
|   | 3.4          |                                         | Data Engine<br>                     | 7  |  |  |  |  |  |  |  |  |  |
|   |              | 3.4.1                                   | OCR 1.0 data<br>                    | 7  |  |  |  |  |  |  |  |  |  |
|   |              | 3.4.2                                   | OCR 2.0 data<br>                    | 8  |  |  |  |  |  |  |  |  |  |
|   |              | 3.4.3                                   | General vision data                 | 9  |  |  |  |  |  |  |  |  |  |
|   |              | 3.4.4                                   | Text-only data                      | 9  |  |  |  |  |  |  |  |  |  |
|   | 3.5          |                                         | Training Pipelines                  | 9  |  |  |  |  |  |  |  |  |  |
|   |              | 3.5.1                                   | Training DeepEncoder<br>            | 10 |  |  |  |  |  |  |  |  |  |
|   |              | 3.5.2                                   | Training DeepSeek-OCR               | 10 |  |  |  |  |  |  |  |  |  |
| 4 |              | Evaluation                              |                                     | 10 |  |  |  |  |  |  |  |  |  |
|   | 4.1          | Vision-text Compression Study<br><br>10 |                                     |    |  |  |  |  |  |  |  |  |  |
|   | 4.2          | OCR Practical Performance<br><br>12     |                                     |    |  |  |  |  |  |  |  |  |  |
|   | 4.3          | Qualitative Study<br>                   |                                     |    |  |  |  |  |  |  |  |  |  |
|   |              | 4.3.1                                   | Deep parsing<br>                    | 12 |  |  |  |  |  |  |  |  |  |
|   |              | 4.3.2                                   | Multilingual recognition            | 16 |  |  |  |  |  |  |  |  |  |
|   |              | 4.3.3                                   | General vision understanding        | 17 |  |  |  |  |  |  |  |  |  |
| 5 |              | Discussion                              |                                     | 18 |  |  |  |  |  |  |  |  |  |
| 6 |              | Conclusion                              |                                     | 19 |  |  |  |  |  |  |  |  |  |

