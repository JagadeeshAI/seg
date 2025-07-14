import torch
from pkgs.UKAN.Seg_UKAN.archs import UKAN

def test_ukan():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UKAN(num_classes=1).to(device)  # Binary segmentation
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Print results
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output min: {output.min().item():.4f}")
    print(f"Output max: {output.max().item():.4f}")
    
    return output

if __name__ == "__main__":
    output = test_ukan()