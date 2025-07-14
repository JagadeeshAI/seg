import torch
from pkgs.PraNetV2.binary_seg.lib.PraNet_Res2Net import PraNet

def test_pranet():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PraNet().to(device)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 352, 352).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Print results
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i+1} shape: {output.shape}")
    
    return outputs

if __name__ == "__main__":
    outputs = test_pranet()