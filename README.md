# LPN: Liquid Probabilistic Network

Core PyTorch model implementation for UAV probabilistic trajectory prediction.

## Model

The model contains a motion feature extraction module and a joint probabilistic
trajectory flow mapping module. A single CfC history layer retains time-indexed
motion information. The decoder combines dual-scale temporal convolution,
history cross-attention, and history-conditioned affine modulation.

The conditional vector field takes normalized motion states. Its public interface
uses the following tensor shapes:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `history` | `[B, 151, 7]` | Observed position, velocity, and turn-rate states |
| `context` | `[B, 8]` | Endpoint position and motion context |
| `state` | `[B, 150, 7]` | Current future-state sample |
| `memory` | `[B, 152, 320]` | Encoded history including the context token |
| Vector field | `[B, 150, 7]` | Update rate of the future-state sample |

The seven state channels are three local position components, three local velocity
components, and turn rate. The context contains three global position components,
cosine and sine of heading, ground speed, vertical velocity, and turn rate.
Observation spacing is 0.2 s. Inputs must already be transformed into the local
heading-aligned frame and standardized with training-set statistics.

## Installation and interface check

```bash
python -m pip install -r requirements.txt
python examples/check_model.py --device cpu
```

The example uses synthetic tensors and randomly initialized weights to check input
and output shapes. It does not produce trained trajectory predictions.

```python
import torch
from lpn import LPN

model = LPN().eval()
history = torch.zeros(1, 151, 7)
context = torch.zeros(1, 8)
state = torch.randn(1, 150, 7)
with torch.no_grad():
    memory = model.encode(history, context)
    update = model.vector_field(state, memory)
```

## CfC dependency and attribution

`lpn/_cfc.py` is the unmodified `CfCCell` implementation from `ncps` version 1.0.1,
copyright Mathias Lechner and Ramin Hasani (2022), distributed under Apache-2.0.
Its copyright header is retained. The license is included in
`third_party/ncps-LICENSE`.

R. Hasani et al., Closed-form continuous-time neural networks,
Nature Machine Intelligence 4 (2022), 992–1003.
https://doi.org/10.1038/s42256-022-00556-7

Upstream implementation: https://github.com/mlech26l/ncps
