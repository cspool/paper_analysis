# LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

Manlai Liang and Jiaming Zhang and Xiong Li and Jinlong Li<sup>∗</sup> AI Lab, China Merchants Bank, China {liangml,zhangjm,lixiong,lucida}@cmbchina.com

#### Abstract

The increasing size of the Key-Value (KV) cache during the Large Language Models longcontext inference is the main obstacle for its balance between the deployment cost and task accuracy. To reduce the KV cache size in such scenarios, most previous efforts leveraged on the attention weight to evict noncritical cache tokens. But there is a tradeoff in those methods, they usually require major modification of the inference infrastructure and significant computation overhead. Based on the fact that the Large Language models are autoregressive models, we propose LagKV, a KV compression strategy only relying on straight forward comparison among KV themselves. It is a totally attention free method which offers easy integration to the main stream inference platform and comparable performance comparing to other complicated KV compression methods. Results on RULER benchmark show that, our approach outperforms SnapKV and StreamingLLM in different compression ratios. Especially in the 64 digit passkey retrieval task, our method outperforms the attention weight based method H2O over 50% with same compression ratios. Our code is available at [https://github.com/](https://github.com/AI-Lab-China-Merchants-Bank/LagKV) [AI-Lab-China-Merchants-Bank/LagKV](https://github.com/AI-Lab-China-Merchants-Bank/LagKV).

