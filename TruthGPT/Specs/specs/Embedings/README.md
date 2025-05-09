



| **Parameter** | **Type** | **Description**                                          |   |
| ------------- | -------- | -------------------------------------------------------- | - |
| `dim`         | `int`    | Embedding dimension (must be even).                      |   |
| `height`      | `int`    | Height of the spatial grid.                              |   |
| `width`       | `int`    | Width of the spatial grid.                               |   |
| `theta`       | `float`  | Scaling factor for frequency decay (default is 10000.0). |   |


🔁 Returns
A tensor of shape (height, width, dim // 2) containing complex exponentials, representing the rotary positional embeddings for a 2D spatial grid.
