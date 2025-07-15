import os
import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate
from thop import profile
from tqdm import tqdm
import warnings
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from data import get_dataloaders
from config import MODEL

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
    
    elif model_type == 'ukan':
        from pkgs.UKAN.Seg_UKAN.archs import UKAN
        checkpoint_path = "checkpoints/best_ukan.pth"
        
        # Create model
        model = UKAN(
            num_classes=1,
            input_channels=3,
            embed_dims=[256, 320, 512],
            no_kan=False
        ).to(DEVICE)
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded U-KAN checkpoint from {checkpoint_path}")
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
        elif model_type == 'ukan':
            # For U-KAN, single output
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

def save_predictions(model, dataloader, model_type, save_dir="results"):
    """Save model predictions and organize results"""
    model.eval()
    
    # Create directories
    rgb_dir = os.path.join(save_dir, "RGB")
    true_mask_dir = os.path.join(save_dir, "true_mask")
    pred_dir = os.path.join(save_dir, model_type)
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(true_mask_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    print(f"Saving predictions for {model_type.upper()}...")
    
    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(tqdm(dataloader, desc=f"Saving {model_type} predictions")):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Get predictions
            outputs = model(images)
            
            # Handle different output formats
            if model_type == 'emcad':
                predictions = torch.sigmoid(outputs[-1])
            elif model_type == 'ukan':
                predictions = torch.sigmoid(outputs)
            else:  # pranetv2
                if isinstance(outputs, (list, tuple)):
                    predictions = torch.sigmoid(outputs[-1])
                else:
                    predictions = torch.sigmoid(outputs)
            
            # Save each image in the batch
            for i in range(images.shape[0]):
                filename = filenames[i] if isinstance(filenames[i], str) else f"image_{batch_idx:04d}_{i:02d}"
                base_name = os.path.splitext(os.path.basename(filename))[0]
                
                # Save RGB image (only once for the first model)
                rgb_path = os.path.join(rgb_dir, f"{base_name}.png")
                if not os.path.exists(rgb_path):
                    # Convert tensor to PIL image (images are already in [0,1] range from ToTensor())
                    rgb_img = images[i].cpu()
                    rgb_img = transforms.ToPILImage()(rgb_img)
                    rgb_img.save(rgb_path)
                
                # Save true mask (only once for the first model)
                mask_path = os.path.join(true_mask_dir, f"{base_name}.png")
                if not os.path.exists(mask_path):
                    true_mask = (masks[i].cpu().squeeze() * 255).numpy().astype(np.uint8)
                    Image.fromarray(true_mask, mode='L').save(mask_path)
                
                # Save prediction
                pred_path = os.path.join(pred_dir, f"{base_name}.png")
                pred_mask = (predictions[i].cpu().squeeze() * 255).numpy().astype(np.uint8)
                Image.fromarray(pred_mask, mode='L').save(pred_path)
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
            elif model_type == 'ukan':
                # U-KAN returns single output
                predictions = torch.sigmoid(outputs)
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
            elif model_type == 'ukan':
                # U-KAN returns single output
                predictions = torch.sigmoid(outputs)
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

def save_comparison_table(results, test_size, save_path="results/comp.txt"):
    """Save comparison table to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    headers = ["Model Name", "Precision", "Recall", "F1 Score", "IoU", "Params (M)", "FLOPs (G)", "FPS", "Test Size"]
    data = []
    
    for result in results:
        data.append([
            result['name'],
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['iou']:.4f}",
            f"{result['params_m']:.2f}",
            f"{result['flops_g']:.2f}" if result['flops_g'] > 0 else "N/A",
            f"{result['fps']:.2f}",
            test_size
        ])
    
    # Generate table string
    table_str = tabulate(data, headers=headers, tablefmt="grid")
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write("="*120 + "\n")
        if len(results) > 1:
            f.write("MODEL COMPARISON RESULTS\n")
        else:
            f.write(f"EVALUATION RESULTS FOR {results[0]['name']}\n")
        f.write("="*120 + "\n")
        f.write(table_str + "\n\n")
        
        # Add detailed breakdown
        for result in results:
            f.write(f"Detailed Results for {result['name']}:\n")
            f.write(f"  • Precision: {result['precision']:.4f}\n")
            f.write(f"  • Recall: {result['recall']:.4f}\n")
            f.write(f"  • F1-Score: {result['f1']:.4f}\n")
            f.write(f"  • IoU: {result['iou']:.4f}\n")
            f.write(f"  • Parameters: {result['total_params']:,} ({result['params_m']:.2f}M)\n")
            f.write(f"  • FLOPs: {result['flops']:,} ({result['flops_g']:.2f}G)\n" if result['flops'] > 0 else f"  • FLOPs: Could not calculate\n")
            f.write(f"  • Inference Speed: {result['fps']:.2f} FPS\n")
            f.write(f"  • Test Samples: {test_size}\n\n")
        
        # Add summary if multiple models
        if len(results) > 1:
            best_iou = max(results, key=lambda x: x['iou'])
            best_fps = max(results, key=lambda x: x['fps'])
            least_params = min(results, key=lambda x: x['params_m'])
            
            f.write("="*60 + "\n")
            f.write("SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Best IoU: {best_iou['name']} ({best_iou['iou']:.4f})\n")
            f.write(f"Fastest: {best_fps['name']} ({best_fps['fps']:.2f} FPS)\n")
            f.write(f"Most Efficient: {least_params['name']} ({least_params['params_m']:.2f}M params)\n")
    
    print(f"Comparison table saved to {save_path}")

def get_model_display_name(model_type):
    """Get display name for model"""
    if model_type == 'emcad':
        return "EMCAD"
    elif model_type == 'pranetv2':
        return "PraNet-V2"
    elif model_type == 'ukan':
        return "U-KAN"
    else:
        return model_type.upper()

def evaluate_single_model(model_type, val_loader, save_predictions_flag=True):
    """Evaluate a single model and return results"""
    try:
        print(f"\nEvaluating {get_model_display_name(model_type)}...")
        
        # Load model
        model = load_model(model_type)
        
        # Save predictions if requested
        if save_predictions_flag:
            save_predictions(model, val_loader, model_type)
        
        print("Calculating performance metrics...")
        # Calculate metrics
        precision, recall, f1, iou = evaluate_model(model, val_loader, model_type)
        
        print("Calculating model complexity...")
        total_params, flops = get_model_complexity(model, model_type)
        
        print("Measuring inference speed...")
        fps = measure_fps(model, val_loader, model_type)
        
        # Format values
        params_m = total_params / 1e6
        flops_g = flops / 1e9 if flops > 0 else 0
        
        return {
            'name': get_model_display_name(model_type),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'params_m': params_m,
            'flops_g': flops_g,
            'fps': fps,
            'total_params': total_params,
            'flops': flops
        }
    except Exception as e:
        print(f"Error evaluating {model_type}: {e}")
        return None

def main():
    """Main function"""
    # Check if model should be processed
    if MODEL is None:
        print("No model specified in config")
        return
    
    model_type = MODEL.lower()
    
    # Load data once
    _, val_loader, _ = get_dataloaders()
    test_size = len(val_loader.dataset)
    
    # Determine which models to evaluate
    if model_type == 'all':
        model_types = ['pranetv2', 'emcad', 'ukan']
        print("Evaluating ALL models...")
    elif model_type in ['pranetv2', 'emcad', 'ukan']:
        model_types = [model_type]
    else:
        print(f"Model {MODEL} not supported. Use 'pranetv2', 'emcad', 'ukan', or 'all'")
        return
    
    # Evaluate models
    results = []
    for mt in model_types:
        result = evaluate_single_model(mt, val_loader, save_predictions_flag=True)
        if result:
            results.append(result)
    
    if not results:
        print("No models successfully evaluated")
        return
    
    # Save comparison table to file
    save_comparison_table(results, test_size)
    
    # Create comparison table for console
    headers = ["Model Name", "Precision", "Recall", "F1 Score", "IoU", "Params (M)", "FLOPs (G)", "FPS", "Test Size"]
    data = []
    
    for result in results:
        data.append([
            result['name'],
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['iou']:.4f}",
            f"{result['params_m']:.2f}",
            f"{result['flops_g']:.2f}" if result['flops_g'] > 0 else "N/A",
            f"{result['fps']:.2f}",
            test_size
        ])
    
    # Print comparison table
    print("\n" + "="*120)
    if len(results) > 1:
        print("MODEL COMPARISON RESULTS")
    else:
        print(f"EVALUATION RESULTS FOR {results[0]['name']}")
    print("="*120)
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Print detailed breakdown for each model
    for result in results:
        print(f"\nDetailed Results for {result['name']}:")
        print(f"  • Precision: {result['precision']:.4f}")
        print(f"  • Recall: {result['recall']:.4f}")
        print(f"  • F1-Score: {result['f1']:.4f}")
        print(f"  • IoU: {result['iou']:.4f}")
        print(f"  • Parameters: {result['total_params']:,} ({result['params_m']:.2f}M)")
        print(f"  • FLOPs: {result['flops']:,} ({result['flops_g']:.2f}G)" if result['flops'] > 0 else f"  • FLOPs: Could not calculate")
        print(f"  • Inference Speed: {result['fps']:.2f} FPS")
        print(f"  • Test Samples: {test_size}")
    
    # Print best model summary if comparing multiple
    if len(results) > 1:
        best_iou = max(results, key=lambda x: x['iou'])
        best_fps = max(results, key=lambda x: x['fps'])
        least_params = min(results, key=lambda x: x['params_m'])
        
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Best IoU: {best_iou['name']} ({best_iou['iou']:.4f})")
        print(f"Fastest: {best_fps['name']} ({best_fps['fps']:.2f} FPS)")
        print(f"Most Efficient: {least_params['name']} ({least_params['params_m']:.2f}M params)")
    
    print(f"\nResults saved in:")
    print(f"  • Comparison table: results/comp.txt")
    print(f"  • RGB images: results/RGB/")
    print(f"  • True masks: results/true_mask/")
    for mt in model_types:
        print(f"  • {get_model_display_name(mt)} predictions: results/{mt}/")

if __name__ == "__main__":
    main()