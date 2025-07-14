import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data import get_dataloaders

# Configuration
try:
    from config import MODEL
except ImportError:
    MODEL = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
LEARNING_RATE = 1e-4
SAVE_DIR = "checkpoints"

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return bce_loss + dice_loss

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU metric"""
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold
    
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def emcad_multi_scale_loss(outputs, targets, criterion, weights=[1.0, 1.0, 1.0, 1.0]):
    """Calculate multi-scale loss for EMCAD (based on paper's Equation 11)"""
    total_loss = 0
    for i, (output, weight) in enumerate(zip(outputs, weights)):
        total_loss += weight * criterion(output, targets)
    
    # Additional combined loss term
    combined_output = sum(outputs) / len(outputs)
    total_loss += criterion(combined_output, targets)
    
    return total_loss

def train_epoch(model, train_loader, criterion, optimizer, device, model_type):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (images, masks, _) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss based on model type
        if model_type == 'emcad':
            # EMCAD returns [p4, p3, p2, p1] - use multi-scale loss
            loss = emcad_multi_scale_loss(outputs, masks, criterion)
            main_output = outputs[-1]  # Use finest scale (p1) for metrics
        elif model_type == 'ukan':
            # U-KAN returns single output
            loss = criterion(outputs, masks)
            main_output = outputs
        else:
            # PraNet-V2 handling
            loss = 0
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    loss += criterion(output, masks)
                main_output = outputs[-1]  # Use final output for metrics
            else:
                loss = criterion(outputs, masks)
                main_output = outputs
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_iou = calculate_iou(main_output, masks)
        
        total_loss += loss.item()
        total_iou += batch_iou
        
        # Update progress
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{batch_iou:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou

def validate_epoch(model, val_loader, criterion, device, model_type):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_iou = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch_idx, (images, masks, _) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss based on model type
            if model_type == 'emcad':
                # EMCAD multi-scale loss
                loss = emcad_multi_scale_loss(outputs, masks, criterion)
                main_output = outputs[-1]
            elif model_type == 'ukan':
                # U-KAN single output
                loss = criterion(outputs, masks)
                main_output = outputs
            else:
                # PraNet-V2 handling
                loss = 0
                if isinstance(outputs, (list, tuple)):
                    for output in outputs:
                        loss += criterion(output, masks)
                    main_output = outputs[-1]
                else:
                    loss = criterion(outputs, masks)
                    main_output = outputs
            
            # Calculate metrics
            batch_iou = calculate_iou(main_output, masks)
            
            total_loss += loss.item()
            total_iou += batch_iou
            
            # Update progress
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{batch_iou:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou

def save_checkpoint(model, optimizer, epoch, iou, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iou': iou,
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_model(model_type):
    """Load model based on type"""
    if model_type == 'emcad':
        from pkgs.EMCAD.lib.networks import EMCADNet
        model = EMCADNet(
            num_classes=1,
            encoder='pvt_v2_b2',
            pretrain=False  # Training from scratch
        ).to(DEVICE)
        print(f"EMCAD model loaded with {sum(p.numel() for p in model.parameters())} parameters")
        
    elif model_type == 'pranetv2':
        from pkgs.PraNetV2.binary_seg.lib.PraNet_Res2Net import PraNet
        model = PraNet().to(DEVICE)
        
        # Try to load pretrained weights if available
        pretrained_path = None
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=DEVICE)
            model.load_state_dict(checkpoint, strict=False)
        else:
            print("No pretrained weights found, using backbone weights only")
    
    elif model_type == 'ukan':
        from pkgs.UKAN.Seg_UKAN.archs import UKAN
        model = UKAN(
            num_classes=1,
            input_channels=3,
            embed_dims=[256, 320, 512],
            no_kan=False  # Use KAN layers
        ).to(DEVICE)
        print(f"U-KAN model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def main():
    """Main training function"""
    # Check if model should be trained
    if MODEL is None:
        return
    
    model_type = MODEL.lower()
    if model_type not in ['pranetv2', 'emcad', 'ukan']:
        print(f"Model {MODEL} not supported. Use 'pranetv2', 'emcad', or 'ukan'")
        return
    
    print(f"Training {MODEL}...")
    
    # Set model-specific paths
    if model_type == 'emcad':
        BEST_MODEL_PATH = "checkpoints/best_emcad.pth"
    elif model_type == 'ukan':
        BEST_MODEL_PATH = "checkpoints/best_ukan.pth"
    else:
        BEST_MODEL_PATH = "checkpoints/best_pranet.pth"
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data
    train_loader, val_loader, _ = get_dataloaders()
    
    # Load model
    model = load_model(model_type)
    
    # Setup training
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
    
    # Training variables
    best_iou = 0.0
    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []
    
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, DEVICE, model_type)
        
        # Validate
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, DEVICE, model_type)
        
        # Update scheduler
        scheduler.step(val_iou)
        
        # Save metrics
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch, val_iou, val_loss, BEST_MODEL_PATH)
            print(f"New best IoU: {best_iou:.4f} - Model saved!")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{SAVE_DIR}/checkpoint_{model_type}_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, val_iou, val_loss, checkpoint_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()