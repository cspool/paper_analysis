# A Datasets

#### A.1 Examples

We provide some examples of the questions and CoT solutions for the datasets used in our experiments.

#### GSM8k

Question = "John cuts his grass to 2 inches. It grows .5 inches per month. When it gets to 4 inches he cuts it back down to 2 inches. It cost \$100 to get his grass cut. How much does he pay per year?"

Steps = ["«4-2=2»", "«2/.5=4»", "«12/4=3»", "«100\*3=300»"]

Answer = "300"

#### ProntoQA

Question = "Brimpuses are not luminous. Shumpuses are amenable. Each yumpus is a lorpus. Gorpuses are shumpuses. Each zumpus is a grimpus. Gorpuses are rompuses. Dumpuses are not floral. Lempuses are cold. Brimpuses are impuses. Every lorpus is floral. Every rompus is transparent. Grimpuses are muffled. Rompuses are yumpuses. Rompuses are wumpuses. Zumpuses are fast. Wumpuses are bitter. Every sterpus is orange. Each lorpus is a vumpus. Yumpuses are feisty. Each yumpus is a lempus. Gorpuses are snowy. Zumpuses are gorpuses. Every lorpus is a sterpus. Stella is a brimpus. Stella is a zumpus. True or false: Stella is not floral."

Steps = ["Stella is a zumpus. Zumpuses are gorpuses.", "Stella is a gorpus. Gorpuses are rompuses.", "Stella is a rompus. Rompuses are yumpuses.", "Stella is a yumpus. Each yumpus is a lorpus.", "Stella is a lorpus. Every lorpus is floral.", "Stella is floral."]

Answer = "False"

### ProsQA

Question = "Every shumpus is a rempus. Every shumpus is a yimpus. Every terpus is a fompus. Every terpus is a gerpus. Every gerpus is a brimpus. Alex is a rempus. Every rorpus is a scrompus. Every rorpus is a yimpus. Every terpus is a brimpus. Every brimpus is a lempus. Tom is a terpus. Every shumpus is a timpus. Every yimpus is a boompus. Davis is a shumpus. Every gerpus is a lorpus. Davis is a fompus. Every shumpus is a boompus. Every shumpus is a rorpus. Every terpus is a lorpus. Every boompus is a timpus. Every fompus is a yerpus. Tom is a dumpus. Every rempus is a rorpus. Is Tom a lempus or scrompus?"

Steps = ["Tom is a terpus.", "Every terpus is a brimpus.", "Every brimpus is a lempus."] Answer = "Tom is a lempus."

#### A.2 Construction of ProsQA

To construct the dataset, we first compile a set of typical entity names, such as "Alex" and "Jack," along with fictional concept names like "lorpus" and "rorpus," following the setting of ProntoQA [\(Saparov and He,](#page-12-5) [2022\)](#page-12-5). Each problem is structured as a binary question: "Is [Entity] a [Concept A] or [Concept B]?" Assuming [Concept A] is the correct answer, we build a directed acyclic graph (DAG) where each node represents an entity or a concept. The graph is constructed such that a path exists from [Entity] to [Concept A] but not to [Concept B].

Algorithm [1](#page-15-0) describes the graph construction process. The DAG is incrementally built by adding nodes and randomly connecting them with edges. To preserve the validity of the binary choice, with some probability, we

| # Nodes | # Edges | Len. of Shortest Path | # Shortest Paths |
|---------|---------|-----------------------|------------------|
| 23.0    | 36.0    | 3.8                   | 1.6              |

Table 2 Statistics of the graph structure in ProsQA.

enforce that the new node cannot simultaneously serve as a descendant to both node 0 and 1. This separation maintains distinct families of nodes and balances their sizes to prevent model shortcuts.

After the graph is constructed, nodes without parents are assigned entity names, while other nodes receive concept names. To formulate a question of the form "Is [Entity] a [Concept A] or [Concept B]?", we designate node 0 in the graph as [Entity], a leaf node labeled 1 as [Concept A], and a leaf node labeled 2 as [Concept B]. This setup ensures a path from [Entity] to [Concept A] without any connection to [Concept B], introducing a moderately complex reasoning path. Finally, to avoid positional biases, [Concept A] and [Concept B] are randomly permuted in each question.

```
Algorithm 1 Graph Construction for ProsQA
```

```
edges ← {}
nodes ← {0, 1}
labels ← {0 : 1, 1 : 2}
                    ▷ Labels: 1 (descendant of node 0), 2 (descendant of node 1), 3 (both), 0 (neither).
groups ← {0 : {}, 1 : {0}, 2 : {1}, 3 : {}}
idx ← 2
while idx < N do
                                       ▷ For each new node, randomly add edges from existing nodes
   n_in_nodes ← poisson(1.5)
   rand ← random()
   if rand ≤ 0.35 then
      candidates ← groups[0] ∪ groups[1] ▷ Cannot be a descendant of node 1.
   else if rand ≤ 0.7 then
      candidates ← groups[0] ∪ groups[2] ▷ Cannot be a descendant of node 0.
   else
      candidates ← nodes
   end if
   n_in_nodes ← min(len(candidates), n_in_nodes)
   weights ← [depth_to_root(c) · 1.5 + 1 ∀c ∈ candidates]
                                                ▷ Define sampling weights to prioritize deeper nodes.
                                     ▷ This way, the solution reasoning chain is expected to be longer.
   in_nodes ← random_choice(candidates, n_in_nodes, prob = weights/sum(weights))
   cur_label ← 0
   for in_idx ∈ in_nodes do
      cur_label ← cur_label | labels[in_idx] ▷ Update label using bitwise OR.
      edges.append((in_idx, idx))
   end for
   groups[cur_label].append(idx)
   labels[idx] ← cur_label
   nodes ← nodes ∪ {idx}
   idx ← idx + 1
end while
```

## A.3 Statistics

We show the size of all datasets in Table [3.](#page-16-1)

| Dataset  | Training | Validation | Test |
|----------|----------|------------|------|
| GSM8k    | 385,620  | 500        | 1319 |
| ProntoQA | 9,000    | 200        | 800  |
| ProsQA   | 17,886   | 300        | 500  |

Table 3 Statistics of the datasets.

