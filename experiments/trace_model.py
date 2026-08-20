import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import torch

from model.build import shvit_s1


def print_shape(name):
    """
    Return a forward hook that prints the output shape of a module.
    """
    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            print(f"{name:<12} -> {tuple(output.shape)}")
        else:
            print(f"{name:<12} -> non-tensor output")
    return hook


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the smallest official SHViT variant.
    model = shvit_s1(num_classes=1000)
    model = model.to(device)
    model.eval()

    # Attach temporary observers to the major stages.
    model.patch_embed.register_forward_hook(print_shape("patch_embed"))
    model.blocks1.register_forward_hook(print_shape("stage1"))
    model.blocks2.register_forward_hook(print_shape("stage2"))
    model.blocks3.register_forward_hook(print_shape("stage3"))
    model.head.register_forward_hook(print_shape("classifier"))

    # One fake RGB image:
    # [batch, channels, height, width]
    x = torch.randn(1, 3, 224, 224, device=device)

    print(f"device       -> {device}")
    print(f"input        -> {tuple(x.shape)}")

    with torch.no_grad():
        output = model(x)

    print(f"final output -> {tuple(output.shape)}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"parameters   -> {num_params:,}")


if __name__ == "__main__":
    main()