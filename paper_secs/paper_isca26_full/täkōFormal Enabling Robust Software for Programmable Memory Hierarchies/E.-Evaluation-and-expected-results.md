# *E. Evaluation and expected results*

The two provided scripts correspond to the 2 components of the artifact. Running the run\_alloy\_tests.sh script will iterate through the litmus tests in the repository and confirm the expected outcomes claimed for these tests in the paper. A full table of these is included below, as well as in the README.

| Figure                        | Claimed Result                               | File (.als)   |  |
|-------------------------------|----------------------------------------------|---------------|--|
| Figure 4a                     | r1 = 2, r2 = 0<br>forbidden                  | test_paper_ex |  |
| Figure 5a                     | r1 = 1, r2 = 0<br>allowed (§V-A)             | test_mp       |  |
| Figure 9a<br>(w/o OnMiss)     | r1 = 1, r2 = 0<br>forbidden                  | test_mp_rmw   |  |
| Figure 9a<br>(w/ OnMiss)      | r1 = 1, r2 = 0<br>forbidden                  | test_mp_rmwcb |  |
| Figure 10a                    | r1 = 0, r2 = 0<br>allowed                    | test_icb_sb   |  |
| Figure 11a<br>(w/o i2))       | racy                                         | test_wbrace   |  |
| Figure 11a<br>(w/ i2))        | a) no race<br>b) r1 = 0<br>forbidden         | test_wbflush  |  |
| Figure 12a<br>(w/ OnWB<br>1)) | racy                                         | test_phir     |  |
| Figure 12a<br>(w/ OnWB<br>2)) | a) no race<br>b) r1 = 0, r2 = 0<br>forbidden | test_phinr    |  |
| Figure 13a<br>(w/o i9))       | racy                                         | test_hatsr    |  |
| Figure 13a<br>(w/ i9))        | a) no race<br>b) r1 ̸= r2<br>forbidden       | test_hatsnr   |  |

Running the run\_dafny\_verification.sh script will verify all the Dafny files that make up the end-to-end proof discussed in §VII. The translation of each axiom presented in Figure 6 to its corresponding file is included in the README.

