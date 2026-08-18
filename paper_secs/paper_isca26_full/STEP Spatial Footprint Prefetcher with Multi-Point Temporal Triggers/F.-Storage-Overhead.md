# *F. Storage Overhead*

We use 8-way set-associative FT/AT/PHT with 256/128/512 total entries, respectively. Table I lists per-entry fields and totals. Compared to address- or page-keyed designs (e.g., SMS [34], Bingo [9], Planaria [27]) that exceed 100 KB, STEP achieves greater improvement with much less storage of 10.5 KB. Relative to simple-event designs (e.g., Gaze [14], PMP [20], DSPatch [12]), STEP spends only a few extra KB to gain early opportunity and later-point accuracy through sequential decisions, resulting in much higher performance.

