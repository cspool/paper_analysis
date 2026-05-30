# 6 Conclusion

In this work, we present EEP, a gradient-free evolutionary search method optimized for pruning within an efficienct parameter space. Through extensive experiments on various downstream datasets, we demonstrate that EEP achieves superior performance and greater sparsity compared to baseline methods. Additionally, we make a novel observation that the performance of SMoE models on downstream tasks can be enhanced through pruning, even without updating the remaining parameters. We discuss the potential reasons for this phenomenon, suggesting that pruning may lead to a more effective routing mechanism by reducing the complexity the router network needs to manage.

Limitations. Although we demonstrated promising results, our approach still requires a potentially costly search process. We leave the optimization of search cost to future work.

