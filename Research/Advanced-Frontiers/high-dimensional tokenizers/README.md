

| **Hyperparameter**                 | **FlowMo (fewer params)** | **FlowMo-Lo** |
|-----------------------------------|----------------------------|----------------|
| Learning rate                     | 0.0001                     | -              |
| Batch size                        | 128                        | -              |
| Weight decay                      | 0                          | -              |
| Num. epochs                       | 40                         | 130            |
| λ<sub>ent</sub>                   | 0.0025                     | -              |
| λ<sub>commit</sub>                | 0.000625                   | -              |
| λ<sub>lpips</sub>                 | 0.1                        | -              |
| Hidden size (µP width)           | 768                        | 1152           |
| MLP ratio                         | 4                          | -              |
| Encoder patch size                | 8                          | 4              |
| Decoder patch size                | 8                          | 4              |
| Encoder depth                     | 8                          | -              |
| Decoder depth                     | 16                         | -              |
| Latent sequence length            | 256                        | -              |
| Latent token size                 | 18                         | 18             |
| Codebook size for entropy loss    | 9                          | 9              |
| Total number of parameters (×10⁶) | 517                        | 945            |



https://arxiv.org/pdf/2503.11056