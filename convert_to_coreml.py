
"""
convert_to_coreml.py

Convert a TorchScript PyTorch model (.pth or .pt) to CoreML format.
Usage: python convert_to_coreml.py --torchscript <path_to_torchscript> --output <output_path> [--quantize]
"""

import argparse
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import numpy as np

def convert_torchscript_to_coreml(torchscript_path, output_path, quantize=False):
    # Load TorchScript model
    model = torch.jit.load(torchscript_path, map_location='cpu')
    model.eval()
    # Try to infer input shape from the model (default to 1x3x224x224 if unknown)
    dummy_input = torch.randn(1, 3, 224, 224)
    input_type = [ct.TensorType(name="image", shape=dummy_input.shape, dtype=np.float32)]
    output_type = [ct.TensorType(name="output", dtype=np.float32)]
    coreml_model = ct.convert(
        model,
        inputs=input_type,
        outputs=output_type,
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
        debug=False
    )
    if quantize:
        coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=16)
    coreml_model.save(output_path)
    print(f"CoreML model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TorchScript PyTorch model to CoreML format.")
    parser.add_argument('--torchscript', type=str, required=True, help='Path to TorchScript .pth or .pt file')
    parser.add_argument('--output', type=str, required=True, help='Output CoreML file path (.mlpackage)')
    parser.add_argument('--quantize', action='store_true', help='Apply 16-bit quantization')
    args = parser.parse_args()
    convert_torchscript_to_coreml(args.torchscript, args.output, args.quantize)
