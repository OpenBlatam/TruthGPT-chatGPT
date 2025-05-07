Biased Opinion Dynamics Model
This module simulates the evolution of opinion distributions over a one-dimensional domain, incorporating both diffusion and nonlocal interactions influenced by biased perception. It captures phenomena such as group polarization and collective drift resulting from individual biases in information gathering.

| Parameter | Description                             | Typical Value |
| --------- | --------------------------------------- | ------------- |
| `D`       | Diffusion coefficient                   | 0.1           |
| `β`       | Bias strength in perception kernel      | 0.5           |
| `σ`       | Standard deviation of perception kernel | 5.0           |
| `dx`      | Spatial discretization step size        | 1.0           |
| `dt`      | Time step size                          | 0.1           |
| `L`       | Number of spatial points                | 100           |
| `T`       | Number of time steps                    | 1000          |

| Parameter | Type    | Description                             | Typical Value |                                 |
| --------- | ------- | --------------------------------------- | ------------- | ------------------------------- |
| `D`       | `float` | Diffusion coefficient                   | 0.1           |                                 |
| `β`       | `float` | Bias strength in perception kernel      | 0.5           |                                 |
| `σ`       | `float` | Standard deviation of perception kernel | 5.0           |                                 |
| `dx`      | `float` | Spatial discretization step size        | 1.0           |                                 |
| `dt`      | `float` | Time step size                          | 0.1           |                                 |
| `L`       | `int`   | Number of spatial points                | 100           |                                 |
| `T`       | `int`   | Number of time steps                    | 1000          | ([arXiv][1], [ResearchGate][2]) |

[1]: https://arxiv.org/pdf/2310.01564?utm_source=chatgpt.com "[PDF] arXiv:2310.01564v3 [physics.soc-ph] 7 Feb 2024"
[2]: https://www.researchgate.net/publication/378810927_Collective_group_drift_in_a_partial-differential-equation-based_opinion_dynamics_model_with_biased_perception_kernels?utm_source=chatgpt.com "Collective group drift in a partial-differential-equation-based opinion ..."
