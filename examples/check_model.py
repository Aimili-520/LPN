"""Check the model interface with synthetic tensors."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lpn import LPN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model = LPN().to(device).eval()
    history = torch.zeros(1, 151, 7, device=device)
    context = torch.zeros(1, 8, device=device)
    state = torch.randn(1, 150, 7, device=device)
    with torch.no_grad():
        memory = model.encode(history, context)
        velocity = model.vector_field(state, memory)
    assert memory.shape == (1, 152, 320)
    assert velocity.shape == state.shape
    assert torch.isfinite(velocity).all()
    print("Model interface check passed")
    print("History memory:", tuple(memory.shape))
    print("Vector field:", tuple(velocity.shape))


if __name__ == "__main__":
    main()
