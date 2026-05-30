# <span id="page-16-0"></span>B ABLATIONS AND ADDITIONAL EXPERIMENTS

In this section, we present results for a number of model ablations that try to identify good choices for the main upcycling algorithm decisions. As mentioned in the main text, unless stated otherwise, vision ablations use a B/16 sparse model with 32 experts, C = 1 and 6 MoE layers placed in the last few block of the model. The dense checkpoint was trained for 14 epochs, and we train for an additional 7 epochs (up to a total of 21 epochs). Note that, for C = 1, comparing performance on a per step basis is a reasonably close approximation of a comparison on a per train time basis.

For our language ablations, our default configuration is unchanged: we use a Base model with 32 experts, C = 2 and 6 MoE layers interspersed throughout the model. We train for between 0.5 million and 1 million extra steps.

