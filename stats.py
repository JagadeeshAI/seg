import os
import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate
from thop import profile
from tqdm import tqdm

from data import get_dataloaders
from config import MODEL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_type):
    """Load model based on type and checkpoint"""
    if model_type == 'emcad':
        from pkgs.EMCAD.lib.networks import EMCADNet
        checkpoint_path = "checkpoints/best_emcad.pth"
        
        # Create model
        model = EMCADNet(
            num_classes=1,
            encoder='pvt_v2_b2',
            pretrain=False
        ).to(DEVICE)
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded EMCAD checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using untrained model")
            
    elif model_type == 'pranetv2':
        from pkgs.PraNetV2.binary_seg.lib.PraNet_Res2Net import PraNet
        checkpoint_path = "checkpoints/best_pranet.pth"
        
        # Create model
        model = PraNet().to(DEVICE)
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded PraNet-V2 checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using untrained model")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU"""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def get_model_complexity(model, model_type, input_size=(1, 3, 320, 320)):
    """Calculate exact parameters and FLOPs"""
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # FLOPs using thop
    dummy_input = torch.randn(input_size).to(DEVICE)
    
    try:
        if model_type == 'emcad':
            # For EMCAD, we need to handle multiple outputs
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        else:
            # For PraNet-V2
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs - {e}")
        flops = 0
    
    return total_params, flops

def measure_fps(model, dataloader, model_type, max_batches=10):
    """Measure inference FPS using real test images"""
    model.eval()
    
    # Collect real images from test loader
    real_images = []
    with torch.no_grad():
        for batch_idx, (images, _, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            real_images.append(images.to(DEVICE))
    
    if not real_images:
        print("Warning: No test images available for FPS measurement")
        return 0.0
    
    # Warmup with real images
    for i in range(min(10, len(real_images))):
        with torch.no_grad():
            _ = model(real_images[i % len(real_images)])
    
    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure inference time on real images
    start_time = time.time()
    total_samples = 0
    
    for images in real_images:
        with torch.no_grad():
            outputs = model(images)
        total_samples += images.size(0)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    if total_time > 0:
        return total_samples / total_time
    else:
        return 0.0

def evaluate_model(model, dataloader, model_type):
    """Evaluate model on test data"""
    all_preds = []
    all_targets = []
    total_iou = 0
    num_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating {model_type.upper()}")
        for images, masks, _ in progress_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            
            # Handle different output formats
            if model_type == 'emcad':
                # EMCAD returns [p4, p3, p2, p1], use finest scale
                predictions = torch.sigmoid(outputs[-1])
            else:
                # PraNet-V2 handling
                if isinstance(outputs, (list, tuple)):
                    predictions = torch.sigmoid(outputs[-1])
                else:
                    predictions = torch.sigmoid(outputs)
            
            # Calculate IoU for each sample
            for i in range(predictions.shape[0]):
                iou = calculate_iou(predictions[i], masks[i])
                total_iou += iou
                num_samples += 1
            
            # Flatten for precision/recall/f1
            pred_flat = (predictions > 0.5).cpu().numpy().flatten()
            target_flat = (masks > 0.5).cpu().numpy().flatten()
            
            all_preds.extend(pred_flat)
            all_targets.extend(target_flat)
            
            progress_bar.set_postfix({"IoU": f"{total_iou/num_samples:.4f}"})
    
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    avg_iou = total_iou / num_samples
    
    return precision, recall, f1, avg_iou

def get_model_display_name(model_type):
    """Get display name for model"""
    if model_type == 'emcad':
        return "EMCAD"
    elif model_type == 'pranetv2':
        return "PraNet-V2"
    else:
        return model_type.upper()

def main():
    """Main function"""
    # Check if model should be processed
    if MODEL is None:
        print("No model specified in config")
        return
    
    model_type = MODEL.lower()
    if model_type not in ['pranetv2', 'emcad']:
        print(f"Model {MODEL} not supported. Use 'pranetv2' or 'emcad'")
        return
    
    print(f"Evaluating {get_model_display_name(model_type)}...")
    
    # Load model and data
    model = load_model(model_type)
    _, val_loader, _ = get_dataloaders()
    
    print("Calculating performance metrics...")
    # Calculate metrics
    precision, recall, f1, iou = evaluate_model(model, val_loader, model_type)
    
    print("Calculating model complexity...")
    total_params, flops = get_model_complexity(model, model_type)
    
    print("Measuring inference speed...")
    fps = measure_fps(model, val_loader, model_type)
    
    test_size = len(val_loader.dataset)
    
    # Format values with units
    params_m = total_params / 1e6
    flops_g = flops / 1e9 if flops > 0 else 0
    
    # Create table
    headers = ["Model Name", "Precision", "Recall", "F1 Score", "IoU", "Params (M)", "FLOPs (G)", "FPS", "Test Size"]
    model_display_name = get_model_display_name(model_type)
    
    data = [[
        model_display_name, 
        f"{precision:.4f}", 
        f"{recall:.4f}", 
        f"{f1:.4f}",
        f"{iou:.4f}", 
        f"{params_m:.2f}", 
        f"{flops_g:.2f}" if flops_g > 0 else "N/A", 
        f"{fps:.2f}", 
        test_size
    ]]
    
    # Print table
    print("\n" + "="*100)
    print(f"EVALUATION RESULTS FOR {model_display_name}")
    print("="*100)
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Print detailed breakdown
    print(f"\nDetailed Results:")
    print(f"  • Model: {model_display_name}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall: {recall:.4f}")
    print(f"  • F1-Score: {f1:.4f}")
    print(f"  • IoU: {iou:.4f}")
    print(f"  • Parameters: {total_params:,} ({params_m:.2f}M)")
    print(f"  • FLOPs: {flops:,} ({flops_g:.2f}G)" if flops > 0 else f"  • FLOPs: Could not calculate")
    print(f"  • Inference Speed: {fps:.2f} FPS")
    print(f"  • Test Samples: {test_size}")

if __name__ == "__main__":
    main()