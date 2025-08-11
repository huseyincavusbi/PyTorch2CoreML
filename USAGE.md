# PyTorch to CoreML Conversion: Step-by-Step Usage Guide

This guide explains how to convert a standard PyTorch model checkpoint (`.pth`) to a CoreML model (`.mlpackage`) using the provided scripts in this repository.

## 1. Prepare Your Environment

- Ensure you have Python 3.9+ and a virtual environment activated.
- Install all dependencies:

```sh
pip install -r requirements.txt
```

## 2. Check If Your Model is TorchScript

To check if your `.pth` or `.pt` file is already a TorchScript model:

```sh
python PyTorch2CoreML/check_torchscript.py --model /path/to/your_model.pth
```
If it is NOT a TorchScript model, continue to the next step.

## 3. Create a Model Definition File

Create a file (e.g., `my_model.py`) with a `get_model()` function that returns your model instance. Example for a timm model:

```python
import timm

def get_model():
    model = timm.create_model('seresnext101_32x8d', pretrained=False, num_classes=7, drop_rate=0.21224189367269414)
    return model
```

## 4. Convert PyTorch Model to TorchScript

Use the script to convert your checkpoint to TorchScript:

```sh
python PyTorch2CoreML/convert_to_torchscript.py \
  --model-def PyTorch2CoreML/my_model.py \
  --weights /path/to/your_model.pth \
  --output /path/to/your_model.pt \
  --input-shape 1 3 224 224
```

## 5. Convert TorchScript Model to CoreML

Convert the TorchScript model to CoreML format:

```sh
python PyTorch2CoreML/convert_to_coreml.py \
  --torchscript /path/to/your_model.pt \
  --output /path/to/your_model.mlpackage
```

## 6. Test the CoreML Model

You can test loading the CoreML model with:

```sh
python -c "import coremltools as ct; model = ct.models.MLModel('/path/to/your_model.mlpackage'); print('Loaded! Inputs:', model.input_description, 'Outputs:', model.output_description)"
```

## Notes
- The `.pth` file must be a state_dict, not a full model object.
- The model definition in `my_model.py` must match the architecture and parameters used during training.
- The scripts support timm models, but you can adapt `my_model.py` for any PyTorch model.
- If you want to add metadata to your CoreML model, you can do so after conversion using `coremltools`.

---

**This workflow takes you from a PyTorch `.pth` checkpoint to a deployable CoreML `.mlpackage` in a reproducible, scriptable way.**
