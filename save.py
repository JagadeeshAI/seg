import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from data import get_dataloaders
from config import MODEL

# Configuration
OUTPUT_DIR = "predictions"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model(model_type):
    """Load trained model from checkpoint based on type"""
    if model_type == 'emcad':
        from pkgs.EMCAD.lib.networks import EMCADNet
        checkpoint_path = "checkpoints/best_emcad.pth"
        
        # Create model
        model = EMCADNet(
            num_classes=1,
            encoder='pvt_v2_b2',
            pretrain=False
        ).to(DEVICE)
        
        model_name = "EMCAD"
        
    elif model_type == 'pranetv2':
        from pkgs.PraNetV2.binary_seg.lib.PraNet_Res2Net import PraNet
        checkpoint_path = "checkpoints/best_pranet.pth"
        
        # Create model
        model = PraNet().to(DEVICE)
        model_name = "PraNet-V2"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {model_name} with IoU: {checkpoint['iou']:.4f}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Using untrained {model_name} model")
    
    model.eval()
    return model, model_name

def tensor_to_image(tensor):
    """Convert tensor to numpy image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        # RGB image
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        # Ensure contiguous array for OpenCV
        img = np.ascontiguousarray(img)
    else:
        # Grayscale mask
        img = tensor.squeeze().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Ensure contiguous array for OpenCV
        img = np.ascontiguousarray(img)
    return img

def create_comparison(rgb_img, true_mask, pred_mask, model_name):
    """Create side-by-side comparison: RGB || True Mask || Pred Mask with labels"""
    h, w = rgb_img.shape[:2]
    true_mask = cv2.resize(true_mask, (w, h))
    pred_mask = cv2.resize(pred_mask, (w, h))
    
    # Ensure all images are contiguous for OpenCV
    rgb_img = np.ascontiguousarray(rgb_img.copy())
    true_mask = np.ascontiguousarray(true_mask.copy())
    pred_mask = np.ascontiguousarray(pred_mask.copy())
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    
    # Add labels to each section
    cv2.putText(rgb_img, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(true_mask, "Ground Truth", (10, 30), font, font_scale, color, thickness)
    cv2.putText(pred_mask, f"{model_name} Prediction", (10, 30), font, font_scale, color, thickness)
    
    comparison = np.hstack([rgb_img, true_mask, pred_mask])
    return comparison

def get_model_predictions(model, images, model_type):
    """Get predictions based on model type"""
    outputs = model(images)
    
    if model_type == 'emcad':
        # EMCAD returns [p4, p3, p2, p1], use finest scale
        predictions = torch.sigmoid(outputs[-1])
    else:
        # PraNet-V2 handling
        if isinstance(outputs, (list, tuple)):
            predictions = torch.sigmoid(outputs[-1])
        else:
            predictions = torch.sigmoid(outputs)
    
    return predictions

def save_predictions(model, dataloader, output_dir, model_type, model_name):
    """Generate and save predictions with comparisons"""
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    total_images = len(dataloader.dataset)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Saving {model_name} predictions", total=len(dataloader))
        
        for batch_idx, (images, masks, filenames) in enumerate(progress_bar):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Get model predictions
            predictions = get_model_predictions(model, images, model_type)
            
            # Apply binary threshold
            predictions = (predictions > 0.5).float()
            
            # Process each image in batch
            for i in range(images.shape[0]):
                # Convert to numpy images
                rgb_img = tensor_to_image(images[i])
                true_mask = tensor_to_image(masks[i])
                pred_mask = tensor_to_image(predictions[i])
                
                # Create comparison
                comparison = create_comparison(rgb_img, true_mask, pred_mask, model_name)
                
                # Save with original filename
                filename = filenames[i]
                output_path = f"{model_output_dir}/{filename}_comparison.png"
                cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                
                # Update progress
                img_idx = batch_idx * dataloader.batch_size + i
                progress_bar.set_postfix({"Images": f"{img_idx + 1}/{total_images}"})

def create_grid_visualization(output_dir, model_type, model_name, max_images=16):
    """Create a grid of comparisons for overview"""
    model_output_dir = os.path.join(output_dir, model_type)
    comparison_files = sorted([f for f in os.listdir(model_output_dir) if f.endswith('_comparison.png')])
    
    if not comparison_files:
        print(f"No comparison images found for {model_name}")
        return
    
    comparison_files = comparison_files[:max_images]
    
    # Calculate grid size
    n = len(comparison_files)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Load first image to get dimensions
    first_img = cv2.imread(os.path.join(model_output_dir, comparison_files[0]))
    h, w = first_img.shape[:2]
    
    # Create grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for idx, filename in enumerate(comparison_files):
        row = idx // cols
        col = idx % cols
        
        img = cv2.imread(os.path.join(model_output_dir, filename))
        if img is not None:
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    # Save grid with model name
    grid_path = os.path.join(model_output_dir, f'{model_type}_grid_overview.png')
    cv2.imwrite(grid_path, grid)
    print(f"Created {model_name} grid overview with {len(comparison_files)} images")

def create_model_comparison(output_dir, filenames, max_comparisons=5):
    """Create side-by-side comparison between different models"""
    available_models = []
    for model_dir in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model_dir)
        if os.path.isdir(model_path):
            available_models.append(model_dir)
    
    if len(available_models) < 2:
        print("Need at least 2 models for comparison")
        return
    
    comparison_dir = os.path.join(output_dir, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Select common files
    common_files = []
    for filename in filenames[:max_comparisons]:
        comparison_file = f"{filename}_comparison.png"
        if all(os.path.exists(os.path.join(output_dir, model, comparison_file)) 
               for model in available_models):
            common_files.append(comparison_file)
    
    print(f"Creating model comparison for {len(common_files)} images")
    
    for filename in common_files:
        model_imgs = []
        for model in available_models:
            img_path = os.path.join(output_dir, model, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Add model name label
                cv2.putText(img, model.upper(), (10, img.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                model_imgs.append(img)
        
        if model_imgs:
            # Stack vertically
            combined = np.vstack(model_imgs)
            output_path = os.path.join(comparison_dir, f"comparison_{filename}")
            cv2.imwrite(output_path, combined)

def main():
    """Main function to run prediction saving"""
    # Check if model should be processed
    if MODEL is None:
        print("No model specified in config")
        return
    
    model_type = MODEL.lower()
    if model_type not in ['pranetv2', 'emcad']:
        print(f"Model {MODEL} not supported. Use 'pranetv2' or 'emcad'")
        return
    
    # Load trained model
    model, model_name = load_trained_model(model_type)
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
    
    print(f"Processing {len(val_loader.dataset)} validation images with {model_name}")
    
    # Save predictions
    save_predictions(model, val_loader, OUTPUT_DIR, model_type, model_name)
    
    # Create grid overview
    create_grid_visualization(OUTPUT_DIR, model_type, model_name)
    
    # Get filenames for potential model comparison
    filenames = []
    for _, _, batch_filenames in val_loader:
        filenames.extend(batch_filenames)
        if len(filenames) >= 10:  # Limit for comparison
            break
    
    # Create model comparison if multiple models exist
    create_model_comparison(OUTPUT_DIR, filenames)
    
    print(f"All {model_name} predictions saved to {OUTPUT_DIR}/{model_type}")
    print("Directory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    └── {model_type}/")
    print(f"        ├── individual_comparisons.png")
    print(f"        └── {model_type}_grid_overview.png")
    if len([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]) > 1:
        print(f"    └── model_comparison/")
        print(f"        └── side_by_side_comparisons.png")

if __name__ == "__main__":
    main()