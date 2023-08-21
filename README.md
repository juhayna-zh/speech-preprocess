# Speech Preprocess

A repo for a solution to denoising and separating for two-speeker-mixed noisy speech, using a [BSRNN](https://arxiv.org/abs/2209.15174) inspired deep learning network.

View demos [here](https://harsh-lawyer-1d0.notion.site/Speech-Preprocess-3d33405d571840148e6b70c87edf3731?pvs=4).

<br/>

### Network Architecture 💡

![Struture](md/structure.png)

### Model Basics ✔️

|*Key*         | *VAlue*                  |
| ----------- | ---------------------- |
| Datasets     | AI Shell-3 & NoiseX-92 |
| FLOPs       | 2.408G                 |
| Weights Size | 61.95M                 |
| Parameters  | 16.15M                 |

<br/>

### Important Metrics 🧭
**Naive Case (only mix, no noise)**

| Metric          | SI-SNR | PESQ(wb) | PESQ(nb) | STOI  |
| --------------- | ------ | -------- | -------- | ----- |
| Raw dataset     | 0.002  | 1.240    | 1.473    | 0.681 |
| BSRNN(modified) | 12.195 | 2.453    | 2.866    | 0.901 |



**Difficult Case (with mix & noise)**

| Metric          | SI-SNR | PESQ(wb) | PESQ(nb) | STOI  |
| --------------- | ------ | -------- | -------- | ----- |
| Raw dataset     | -0.597 | 1.146    | 1.379    | 0.656 |
| BSRNN(modified) | 11.384 | 2.212    | 2.661    | 0.880 |

<br/>

### Training Visualization 📉
![Train](md/train.png)
