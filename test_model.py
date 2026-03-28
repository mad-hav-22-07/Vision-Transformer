"""Quick test: verify LDTR-inspired model forward pass shapes."""
import torch
import sys
sys.path.insert(0, ".")
from src.model import ViTLaneSeg

print("Creating LDTR-inspired model...")
model = ViTLaneSeg(
    img_size=(360, 640),
    backbone_channels=(64, 128, 256),
    embed_dim=256,
    num_heads=8,
    num_classes=3,
    transformer_depth=6,
    num_decoder_layers=6,
    num_queries=100,
    num_deform_points=4,
    fpn_channels=128,
)

# --- Training mode ---
model.train()
print("\nRunning forward pass in TRAINING mode (batch_size=2)...")
x = torch.randn(2, 3, 360, 640)
with torch.no_grad():
    logits, heatmap = model(x)

print(f"Input shape:    {x.shape}")
print(f"Logits shape:   {logits.shape}")
print(f"Heatmap shape:  {heatmap.shape}")

expected_logits = torch.Size([2, 3, 360, 640])
expected_heatmap = torch.Size([2, 1, 360, 640])
assert logits.shape == expected_logits, f"Logits shape mismatch! Got {logits.shape}"
assert heatmap.shape == expected_heatmap, f"Heatmap shape mismatch! Got {heatmap.shape}"
print("Training mode shapes: PASSED ✓")

# --- Evaluation mode ---
model.eval()
print("\nRunning forward pass in EVAL mode (batch_size=2)...")
with torch.no_grad():
    out = model(x)

# In eval mode, model returns only logits (not tuple)
assert not isinstance(out, tuple), f"Eval should return tensor, got tuple"
print(f"Output shape:   {out.shape}")
assert out.shape == expected_logits, f"Shape mismatch! Got {out.shape}"
print("Eval mode shapes: PASSED ✓")

params = model.get_param_count()
print(f"\nEncoder params: {params['encoder']:,}")
print(f"Decoder params: {params['decoder']:,}")
print(f"Total:          {params['total_M']}")
print("\nAll forward pass tests PASSED! ✓")
