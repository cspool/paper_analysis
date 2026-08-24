# E.3 Why LightThinker generates more tokens with smaller cache size?

As shown in Figure [4\(](#page-6-0)e-f), we find that Light-Thinker generates more tokens with smaller cache size. We examined outputs under different cache sizes and found that when the cache size is small, the model tends to repeat previous content more often. We believe this is because smaller cache sizes lead to greater information loss during compression, prompting the model to regenerate earlier content more frequently to retain as much information as possible.

