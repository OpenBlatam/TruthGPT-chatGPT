



# TABLE I  
**APPLICATIONS OF LLMS IN SCIENTIFIC FIELDS AND CORRESPONDING TRAINING OPTIMIZATION STRATEGIES**

| Field        | Work               | Backbone            | Main Building Block | #Parameters        | Memory Cost (est.) | Optimizations                                                                                 | Tasks                                                                 |
|--------------|--------------------|---------------------|---------------------|--------------------|---------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **Biology**  | AlphaFold 2 [2]    | -                   | Evoformer [2]       | 93M                | 1.45 GB             | DP, mixed-precision, GC                                                                       | protein structure prediction                                          |
|              | RosettaFold [45]   | -                   | SE(3)-Transformer [46] | 130M           | 2.03 GB             | DP, GA                                                                                        | protein structure prediction                                          |
|              | AlphaFold 3 [47]   | -                   | Pairformer [47]     | Unreported         | -                   | Unreported                                                                                    | biomolecular complex structure prediction                             |
|              | OpenFold [48]      | -                   | Evoformer           | 93M                | 1.45 GB             | DP (ZeRO-2), mixed-precision, GC, offloading, GA                                              | protein structure prediction                                          |
|              | FastFold [49]      | -                   | Evoformer           | 93M                | 1.45 GB             | DP, DAP                                                                                       | protein structure prediction                                          |
|              | ScaleFold [50]     | -                   | Evoformer           | 97M                | 1.52 GB             | DP, DAP                                                                                       | protein structure prediction                                          |
|              | ESMFold [51]       | ESM-2               | Transformer [17]    | 15B                | 240 GB              | DP (FSDP)                                                                                     | protein structure prediction                                          |
|              | xTrimoPGLM [52]    | xTrimoPGLM-100B     | Transformer         | 100B               | 1.56 TB             | DP (ZeRO-1), PP (1F1B), TP (Megatron-LM), mixed-precision, GC                                 | protein understanding, protein generation                             |
|              | ESM3 [53]          | -                   | Transformer         | 1.4B / 7B / 98B     | 1.53 TB             | DP (FSDP), mixed-precision                                                                    | protein reasoning, protein generation                                 |

| **Medicine** | BioGPT [54]        | GPT-2 Medium [31]   | Transformer         | 347M               | 5.42 GB             | DP, GA                                                                                        | biomedical text understanding, biomedical text generation             |
|              | Med-PaLM [3]       | PaLM 540B [34]      | Transformer         | 540B               | 8.44 TB             | DP (ZeRO-3), TP, GC                                                                           | medical question answering, medical reasoning                         |
|              | Med-PaLM 2 [4]     | PaLM 2 340B [35]    | Transformer         | 340B               | 5.31 TB             | Unreported                                                                                    | medical question answering, medical reasoning                         |
|              | Med-PaLM M [5]     | PaLM-E [55]         | Transformer         | 12B / 84B / 562B   | 8.78 TB             | DP (ZeRO-3), TP, GC                                                                           | biomedical generalist                                                 |
|              | BiomedGPT [56]     | OFA [57]            | Transformer         | 33M / 93M / 182M   | 2.84 GB             | DP (PyTorch DDP), mixed-precision                                                             | biomedical generalist                                                 |
|              | Meditron [58]      | Llama-2 [19]        | Transformer         | 7B / 70B           | 1.09 TB             | DP, PP, TP (Megatron-LM)                                                                      | medical reasoning                                                     |
|              | HuatuoGPT [59]     | Baichuan-7B / Ziya-LLaMA-13B [60] | Transformer | 7B / 13B           | 208 GB              | DP (ZeRO)                                                                                     | medical consultation                                                  |
|              | HuatuoGPT-II [61]  | Baichuan2 [62] / Yi-34B [63] | Transformer   | 7B / 13B / 34B     | 544 GB              | DP (ZeRO)                                                                                     | medical consultation                                                  |

| **Biomedicine** | PharmBERT [64]  | BERT-Base [29]      | Transformer         | 110M               | 1.72 GB             | Unreported                                                                                    | drug labeling                                                         |
|                 | PharmGPT [65]   | Llama-2             | Transformer         | 3B / 13B / 70B     | 1.09 TB             | DP+PP+TP                                                                                      | text understanding, text generation                                   |

| **Chemistry** | ChemBERT [66]     | BERT-Base           | Transformer         | 110M               | 1.72 GB             | Unreported                                                                                    | product extraction, reaction role labeling                            |
|               | CatBERTa [8]      | RoBERTa [67]        | Transformer         | 355M               | 5.55 GB             | Unreported                                                                                    | catalyst property prediction                                          |
|               | Chemformer [9]    | BART [68]           | Transformer         | 45M / 230M         | 3.59 GB             | DP (ZeRO-2), mixed-precision                                                                  | reaction prediction, molecular optimization, molecular property prediction |
|               | MolGen [7]        | BART                | Transformer         | 355M               | 5.55 GB             | DP (ZeRO-2), mixed-precision, GA                                                              | molecule generation                                                   |
|               | ChemGPT [6]       | GPT-Neo [69]        | Transformer         | 1.2B               | 19.20 GB            | DP (PyTorch DDP)                                                                              | molecule generation                                                   |

| **Meteorology** | Pangu-Weather [11] | -                | Swin transformer [70] | 256M             | 4 GB                | DP                                                                                           | weather forecast                                                      |
|                 | FuXi [12]         | -                  | Swin transformer [71] | 4.5B             | 72 GB               | DP (FSDP), mixed-precision, GC                                                                 | weather forecast                                                      |
|                 | ClimaX [13]       | -                  | ViT [72]             | Unreported         | -                   | DP, mixed-precision                                                                           | weather forecast, climate projection                                  |
|                 | Aurora [14]       | -                  | Swin transformer     | 1.3B               | 20.80 GB            | DP, mixed-precision, GC                                                                       | atmospheric prediction                                                |

> **Abbreviations**:  
> DP: Data Parallelism  
> PP: Pipeline Parallelism  
> TP: Tensor Parallelism  
> DAP: Data + Activation Parallelism  
> FSDP: Fully Sharded Data Parallel  
> GC: Gradient Checkpointing  
> GA: Gradient Accumulation  





## Suvery 
Memory 
https://arxiv.org/pdf/2501.11847
