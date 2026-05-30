# 5 The Learned Group Structures

What do the groups discover? It is an interesting question, because the group training is end to end and no enforcement of group structure is used. Regardless, we found there are linguistic structures in the learned groups. When trained with Sinkhorn normalization (K=8, τ=0.1), centroids discover interpretable linguistic categories without supervision:

| Group | Category               | Top tokens                               |
|-------|------------------------|------------------------------------------|
| G4    | Punctuation (96% pure) | , (×55), . (×24), ; (×4), – (×7)         |
| G3    | Determiners            | the (×38), a (×14), this (×5), my (×3)   |
| G0    | Prepositions           | to (×14), of (×14), in (×13), for (×5)   |
| G5    | Connectives            | who (×7), which (×7), and (×6), but (×5) |
| G7    | Verbs + pronouns       | have (×6), are (×5), is (×4), I (×4)     |
| G1    | Content/nouns          | Nature, freedom, Land, sense, home       |

Assignment confidence is high (avg 0.89) and groups are balanced (10–16% each). These categories persist through fine-tuning of all 124M parameters. Notably, prepositions and determiners form separate groups—traditional POS tagging lumps them together as "function words," but Focus discovers they serve different attention roles: determiners point to their noun; prepositions link phrases across distance.

Long-range pairing examples. The learned groups enable same-group tokens to attend across long distances. Here are concrete examples from a PG-19 passage: 'Henry' (pos 18) → 'Walker' (pos 772), distance 754, group affinity 0.945 (entity tracking); 'When' (pos 2) → 'since' (pos 390), affinity 0.988 (temporal connectives). These groupings emerge end-to-end from the language modeling objective alone—no supervision on group semantics is provided. Focus discovers these groupings and uses the learned structure to determine which token pairs attend at long range.

