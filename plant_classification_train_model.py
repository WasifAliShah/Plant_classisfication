import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# Configuration
SAMPLE_SIZE_PER_CLASS = 500  # Sample size per class to keep training time reasonable
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_RATIO = 0.7

# Define class names and their folder paths
CLASS_MAPPING = {
    'Apple': ['Apple-20251205T193548Z-3-001/Apple'],
    'Corn': ['Corn-20251205T165237Z-3-001/Corn'],
    'Grape': ['Grape-20251205T165237Z-3-001/Grape', 'Grape-20251205T165237Z-3-002/Grape'],
    'Peach': ['Peach-20251205T165239Z-3-001/Peach'],
    'Potato': ['Potato-20251205T204200Z-3-001/Potato', 'Potato-20251205T204200Z-3-002/Potato'],
    'Strawberry': ['Strawberry-20251205T194622Z-3-001/Strawberry', 'Strawberry-20251205T204209Z-3-001/Strawberry'],
    'Tomato': ['Tomato-20251205T204228Z-3-001/Tomato', 'Tomato-20251205T204228Z-3-002/Tomato', 
               'Tomato-20251205T204228Z-3-003/Tomato', 'Tomato-20251205T204228Z-3-004/Tomato']
}

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (224, 224))
            if self.transform:
                image = self.transform(image)
            return image, label

def collect_images():
    """Collect and sample images from each class"""
    all_images = []
    all_labels = []
    class_names = sorted(CLASS_MAPPING.keys())
    
    print("Collecting images from each class...")
    for class_idx, class_name in enumerate(class_names):
        class_images = []
        folders = CLASS_MAPPING[class_name]
        
        for folder in folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        class_images.append(os.path.join(folder, file))
        
        # Sample images if we have more than SAMPLE_SIZE_PER_CLASS
        if len(class_images) > SAMPLE_SIZE_PER_CLASS:
            class_images = random.sample(class_images, SAMPLE_SIZE_PER_CLASS)
        elif len(class_images) == 0:
            print(f"  Warning: No images found for {class_name}")
            continue
        
        all_images.extend(class_images)
        all_labels.extend([class_idx] * len(class_images))
        print(f"  {class_name}: {len(class_images)} images")
    
    return all_images, all_labels, class_names

def create_data_loaders(image_paths, labels, class_names):
    """Create train and test data loaders"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=1-TRAIN_RATIO, random_state=42, stratify=labels
    )
    
    print(f"\nTrain set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PlantDataset(X_train, y_train, transform=train_transform)
    test_dataset = PlantDataset(X_test, y_test, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, class_names

def create_alexnet_model(num_classes):
    """Create AlexNet model with transfer learning"""
    # try:
        # Try new API first (torchvision >= 0.13)
        # model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    # except (AttributeError, TypeError):
        # Fallback to old API
    model = models.alexnet(pretrained=True)

   # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier - AlexNet uses adaptive pooling, so input is 256*6*6 = 9216
    # But we need to check the actual output size
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(9216, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes)
    )
    
    return model

def create_vggnet_model(num_classes):
    """Create VGGNet model with transfer learning"""

    model = models.vgg16(pretrained=True)

    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes)
    )
    
    return model

def create_resnet50_model(num_classes):
    """Create ResNet50 model with transfer learning"""
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_model(model, train_loader, test_loader, model_name, num_classes):
    """Train the model and return history"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(test_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
    
    return model, history

def plot_learning_curves(history, model_name):
    """Plot learning curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {model_name}_learning_curves.png")

def plot_confusion_matrix(model, test_loader, class_names, model_name):
    """Generate and plot confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {model_name}_confusion_matrix.png")
    
    # Print classification report
    print(f"\n{classification_report(all_labels, all_preds, target_names=class_names)}")
    
    return cm

def main():
    print("Plant Species Classification using Transfer Learning")
    print("="*60)
    
    # Collect and organize data
    image_paths, labels, class_names = collect_images()
    num_classes = len(class_names)
    
    print(f"\nTotal images collected: {len(image_paths)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {', '.join(class_names)}")
    
    # Create data loaders
    train_loader, test_loader, class_names = create_data_loaders(image_paths, labels, class_names)
    
    # Models to train
    models_to_train = {
        'AlexNet': create_alexnet_model(num_classes),
        'VGGNet': create_vggnet_model(num_classes),
        'ResNet50': create_resnet50_model(num_classes)
    }
    # Train each model
    trained_models = {}
    histories = {}
    
    for model_name, model in models_to_train.items():
        trained_model, history = train_model(model, train_loader, test_loader, model_name, num_classes)
        trained_models[model_name] = trained_model
        histories[model_name] = history
        
        # Plot learning curves
        plot_learning_curves(history, model_name)
        
        # Generate confusion matrix
        plot_confusion_matrix(trained_model, test_loader, class_names, model_name)
        
        # Save model
        model_path = f'{model_name}_plant_classifier.pth'
        torch.save(trained_model.state_dict(), model_path)
        print(f"Saved model to {model_path}\n")
    
    print("Training completed successfully!")
    print("\nGenerated files:")
    print("  - Learning curves: *_learning_curves.png")
    print("  - Confusion matrices: *_confusion_matrix.png")
    print("  - Trained models: *.pth")

if __name__ == '__main__':
    main()