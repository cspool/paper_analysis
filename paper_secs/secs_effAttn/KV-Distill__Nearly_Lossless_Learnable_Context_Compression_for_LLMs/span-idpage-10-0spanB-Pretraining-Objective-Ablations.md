# <span id="page-10-0"></span>B Pretraining Objective Ablations

We assess the necessity of both the forward and reverse terms in the loss by evaluating SQuAD performance on multiple different pre-training losses with varying λ values in Equation [6.](#page-4-1) In Table [5](#page-10-1) we observe that using either the pure forward or reverse divergences performs markedly worse than using a mixture of both. Furthermore, using solely the auto-encode + cross-entropy (used in ICAE and DODO), performs significantly worse than Equation [6,](#page-4-1) demonstrating the significant benefits that the weighted distillation objective provide. We also note that replacing the routing mechanism with a learnable embedding (added to important tokens) does not perform well.

<span id="page-10-1"></span>

| Loss       | SQuAD Acc. (%) |  |
|------------|----------------|--|
| λ = 1      | 83.4%          |  |
| λ = 0.6    | 86.0%          |  |
| λ = 0.4    | 85.3%          |  |
| λ = 0      | 82.7%          |  |
| AE + LM    | 79.1%          |  |
| NO ROUTING | 67.4 %         |  |

Table 5: Effect of Pretraining Loss wrt LLAMA-3.

### C Qualitative Analysis of Generated Text

In Figure [6](#page-10-2) we present representative summaries of the short story "Galactic Ghost". LLAMA-3 was prompted to summarize the short story in a few sentences. As the KV retention ratio decreases, we observe an increase in statements that are unfaithful to the original context. Particularly, at extremely high compression ratios, the model begins to hallucinate details, but impressively still maintains understanding of the general plot of the story. Note the varying degrees of *semantic compression* in summaries generated under high compression ratios; qualitatively, we observe that models with severely compressed contexts generally understand the text, but resort to making vague statements.

#### (a) 100% KV retention

<span id="page-10-2"></span>The story follows Willard, a space explorer who is left alone after his friend and space mate, Dobbin, dies. Willard is haunted by the memory of Dobbin's final words, "I saw the Ghost Ship," and becomes obsessed with finding the ship. He spends years searching for it, and eventually, he sees a strange, ghostly ship that seems to be heading towards him. As he gets closer, he realizes that it's not just a hallucination, but a real ship from Earth that's been traveling through space for centuries. In the end, Willard is taken aboard the ship, but learns that the ship and its crew are mere shells of their former selves, and that they are doomed to remain in space forever. Ultimately, Willard is forced to confront the reality that he'll never return to Earth.

#### (b) 2.5% KV retention

The story follows the journey of Willard, a space traveler who has been alone in space for years after the death of his friend and partner, Dobbin. Willard is struggling to come to terms with his isolation and the loss of his friend, and is haunted by the legend of the Ghost Ship, a mythical vessel that is said to roam the solar system, never to be seen again. One day, Willard spots a strange, rocket-shaped object approaching him, and as it draws closer, he realizes it is the Mary Lou. The ship is now a mere shadow of its former self, and Willard is invited to board it, where he is greeted by a rescue team who have been sent to retrieve him. As he adjusts to life on the rescue ship, Willard comes to terms with his past and the reality of his situation, and ultimately accepts that he will never return to Earth. The story concludes with Willard's poignant realization that he and the crew of the rescue ship are the Ghosts, doomed to roam the solar system forever.

### (c) .25% KV retention

The story follows the journey of Willard, a space explorer who has been traveling through the galaxy for years, searching for a new home for humanity. He has lost his friend and companion, Dobbin, and is struggling to come to terms with his own mortality. One day, he discovers a faint light in the distance, which he believes to be a sign of life. As he approaches the light, he realizes it is a ghost ship, a ship that has been drifting through space for millions of years, its crew and passengers long dead. The ship is a relic of a bygone era, and Willard is forced to confront the reality that he will never be able to return to Earth, as the ship would pass through it, making it impossible to survive. In the end, Willard comes to accept his fate, realizing that he and the crew of the ghost ship are doomed to drift through space forever, a reminder of the transience of human existence.

Figure 6: LLAMA-3 was tasked with summarizing a 6k token short story at low KV retention rates. Inaccuracies in the summary are highlighted yellow, and were determined by hand.