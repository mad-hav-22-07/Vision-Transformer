"""Quick test: verify model forward pass shapes."""
import torch
import sys
sys.path.insert(0, ".")
from src.model import ViTLaneSeg

print("Creating model...")
model = ViTLaneSeg(
    img_size=(360, 640),
    patch_size=16,
    embed_dim=512,
    depth=12,
    num_heads=8,
    num_classes=3,
)

print("Running forward pass (batch_size=2)...")
x = torch.randn(2, 3, 360, 640)
with torch.no_grad():
    out = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
expected = torch.Size([2, 3, 360, 640])
print(f"Expected:     {expected}")
assert out.shape == expected, f"Shape mismatch! Got {out.shape}"

params = model.get_param_count()
print(f"\nEncoder params: {params['encoder']:,}")
print(f"Decoder params: {params['decoder']:,}")
print(f"Total:          {params['total_M']}")
print("\nForward pass test PASSED!")
