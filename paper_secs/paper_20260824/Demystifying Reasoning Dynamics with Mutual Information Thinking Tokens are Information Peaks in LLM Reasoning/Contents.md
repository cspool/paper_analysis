# Contents

| 1 | Introduction                                                                           | 1  |  |  |  |  |  |
|---|----------------------------------------------------------------------------------------|----|--|--|--|--|--|
| 2 | Emergence of MI Peaks in LRMs' Reasoning Trajectories                                  |    |  |  |  |  |  |
|   | 2.1<br>Preliminaries<br>                                                               | 3  |  |  |  |  |  |
|   | 2.2<br>Investigating LRM's Reasoning Trajectories with MI<br>                          | 4  |  |  |  |  |  |
|   | 2.3<br>Theoretical Insights: Higher MI Leads to Tighter Bounds on Prediction Error<br> | 4  |  |  |  |  |  |
|   | 2.4<br>Will Non-reasoning LLMs also Exhibit the MI Peaks Phenomenon?<br>               | 5  |  |  |  |  |  |
| 3 | Thinking Tokens are Information Peaks in LLM Reasoning                                 | 6  |  |  |  |  |  |
|   | 3.1<br>Exploring MI Peak Representations in Token Space                                | 6  |  |  |  |  |  |
|   | 3.2<br>Tokens at MI Peaks are Critical to LRM's Reasoning Performance<br>              | 7  |  |  |  |  |  |
| 4 | Applications: Leveraging MI Peaks to Improve LRM Reasoning                             |    |  |  |  |  |  |
|   | 4.1<br>Recycling High-MI Representations During Inference                              | 8  |  |  |  |  |  |
|   | 4.2<br>Test-Time Scaling with Thinking Tokens                                          | 8  |  |  |  |  |  |
| 5 | Related work                                                                           | 9  |  |  |  |  |  |
| 6 | Conclusion                                                                             | 9  |  |  |  |  |  |
| A | Proofs and Definitions                                                                 | 15 |  |  |  |  |  |
|   | A.1<br>Proof of Theorem 1                                                              | 15 |  |  |  |  |  |
|   | A.2<br>Proof of Theorem 2                                                              | 16 |  |  |  |  |  |
|   | A.3<br>Definitions                                                                     | 17 |  |  |  |  |  |
| B | Experimental Implementation Details                                                    | 17 |  |  |  |  |  |
| C | Discussions                                                                            | 18 |  |  |  |  |  |
| D | Additional Experimental Results                                                        | 19 |  |  |  |  |  |

