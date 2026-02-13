# Vision Module

CNN-based intrinsic condition encoder for extracting interpretable visual features from Pokémon card images.

## Overview

The vision module reframes **grading as a feature extraction problem** rather than direct grade prediction. It extracts visual signals aligned with PSA/BGS grading logic:

- **Centering**: Pixel-perfect border symmetry analysis
- **Edge Wear**: Edge sharpness and degradation detection
- **Corner Quality**: Corner wear quantification
- **Surface Condition**: Surface wear, print defects, micro-scratches

## Architecture

```
Input Image (224x224x3)
         ↓
   ResNet50 Backbone (pretrained)
         ↓
   Custom Head (FC layers + Dropout)
         ↓
   Multi-task Outputs:
   ├── Centering Score (0-10)
   ├── Edge Score (0-10)
   ├── Corner Score (0-10)
   ├── Surface Score (0-10)
   └── Condition Embedding (256-dim)
```

## Key Features

- **Interpretable Features**: Sub-scores mirror human grading logic
- **Transfer Learning**: Uses pretrained backbones (ResNet, EfficientNet)
- **Attention Mechanisms**: Grad-CAM visualizations show what model focuses on
- **Robust to Variations**: Handles different lighting, angles, image quality

## Usage

### Training

```bash
python scripts/training/train_vision.py --config configs/vision/cnn_config.yaml
```

### Inference

```python
from src.vision_module.model import VisionEncoder
from src.vision_module.feature_extractor import extract_features

# Load model
model = VisionEncoder.from_pretrained('models/vision/best_model.pth')

# Extract features
embedding, subscores = extract_features(
    model=model,
    image_path='path/to/card.jpg'
)

print(f"Condition Embedding: {embedding.shape}")  # (256,)
print(f"Sub-scores: {subscores}")  # Dict with centering, edge, corner, surface
```

### Interpretability

```python
from src.vision_module.interpretability import generate_gradcam

# Generate attention map
gradcam_map = generate_gradcam(
    model=model,
    image_path='path/to/card.jpg',
    target_layer='layer4'
)
```

## Files

- `model.py`: CNN architecture definition
- `data_loader.py`: Image preprocessing and data loading
- `train.py`: Training loop with validation
- `feature_extractor.py`: Embedding extraction utilities
- `interpretability.py`: Grad-CAM and saliency visualization

## Evaluation Metrics

- **Grade Correlation**: Pearson/Spearman correlation with PSA grades
- **Sub-score Alignment**: Agreement with human grading breakdown
- **Robustness**: Performance across lighting/angle variations
- **Interpretability Quality**: Grad-CAM alignment with visual defects

## References

- Selvaraju et al. (2017): Grad-CAM: Visual Explanations from Deep Networks
- PSA Grading Standards: https://www.psacard.com/resources/gradingstandards
