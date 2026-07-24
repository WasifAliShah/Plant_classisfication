# Plant Species Classification using Transfer Learning

This project implements transfer learning on three CNN architectures (AlexNet, VGGNet, and ResNet50) for classifying 7 different plant species from leaf images.

## Dataset

The dataset contains 7 classes of plant species:
- Apple
- Corn
- Grape
- Peach
- Potato
- Strawberry
- Tomato

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python plant_classification_transfer_learning.py
```

## Features

1. **Data Sampling**: Automatically samples 500 images per class (configurable) to keep training time reasonable
2. **Data Split**: 70% training, 30% testing with stratified splitting
3. **Transfer Learning**: 
   - Freezes pre-trained weights
   - Replaces classification layers for 7-class classification
   - Trains only the new classification layers
4. **Three Models**:
   - AlexNet
   - VGGNet (VGG16)
   - ResNet50
5. **Training Configuration**:
   - Batch size: 32
   - Learning rate: 0.001
   - Optimizer: Adam
   - Epochs: 10
   - Learning rate scheduler: StepLR (reduces LR by 0.1 every 5 epochs)
6. **Outputs**:
   - Learning curves (training/validation loss and accuracy) for each model
   - Confusion matrices for each model
   - Saved trained models (.pth files)
   - Classification reports

## Output Files

After running the script, you'll get:
- `AlexNet_learning_curves.png`
- `VGGNet_learning_curves.png`
- `ResNet50_learning_curves.png`
- `AlexNet_confusion_matrix.png`
- `VGGNet_confusion_matrix.png`
- `ResNet50_confusion_matrix.png`
- `AlexNet_plant_classifier.pth`
- `VGGNet_plant_classifier.pth`
- `ResNet50_plant_classifier.pth`

## Configuration

You can modify the following parameters in the script:
- `SAMPLE_SIZE_PER_CLASS`: Number of images to sample per class (default: 500)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `NUM_EPOCHS`: Number of training epochs (default: 10)
- `TRAIN_RATIO`: Training data ratio (default: 0.7)

## Notes

- The script automatically detects available GPU and uses it if available
- Data augmentation is applied to training data (random flip, rotation, color jitter)
- All models use ImageNet pre-trained weights
- The script handles multiple folders for the same class (e.g., multiple Tomato folders)

