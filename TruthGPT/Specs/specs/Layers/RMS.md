
RMS Layer
This module defines the RMS (Root Mean Square) Layer, commonly used for feature scaling in deep learning models, especially in layers where normalization is crucial for stable training.

📥 Parameters


| Parameter        | Type    | Description                                                         |
| ---------------- | ------- | ------------------------------------------------------------------- |
| `input_dim`      | `int`   | The dimensionality of the input features.                           |
| `scaling_factor` | `float` | A scaling factor to adjust the normalized output (default: `1.0`).  |
| `epsilon`        | `float` | A small constant added to avoid division by zero (default: `1e-8`). |

Input / Output Shape
Input: (batch_size, seq_len, input_dim)

Output: (batch_size, seq_len, input_dim)

