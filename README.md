# PyTorch2CoreML

Convert trained PyTorch models to Apple's CoreML format for iOS/macOS deployment.

## Notebooks

- **`pytorch_to_coreml_conversion.ipynb`** - Complete conversion pipeline from PyTorch to CoreML
- **`core_ml_inference.ipynb`** - Run inference with converted CoreML models on Apple Silicon

## Quick Start

```bash
# Install dependencies
uv sync

# Run notebook in VS Code or
uv run jupyter notebook pytorch_to_coreml_conversion.ipynb
```

## Requirements

- macOS 
- Python 3.9+
- Dependencies: PyTorch, CoreMLTools, torchvision
- Pre-trained model: `best_seresnext50_model.pth`

## What's Included

**Conversion Pipeline:**
- Example SE-ResNeXt50 architecture recreation
- PyTorch model loading and validation
- TorchScript intermediate conversion
- CoreML format conversion

**Inference Pipeline:**
- CoreML model loading and testing
- Medical image preprocessing
- Real-time inference on Apple Silicon
- Performance benchmarking
- Batch prediction and visualization

## Use Cases

- Deploy PyTorch models on iOS/macOS
- Convert medical image classifiers
- Test CoreML inference performance
- Optimize models for Apple devices
- Create mobile-ready ML apps
- Run real-time medical image analysis

## Output

- **Standard Model**: `seresnext50_mri_coreml.mlpackage`
