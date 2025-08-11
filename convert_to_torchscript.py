"""
convert_to_torchscript.py

Convert a standard PyTorch model (with weights) to TorchScript format.
Usage: python convert_to_torchscript.py --model-def <model_def.py> --weights <weights.pth> --output <output.pt> [--input-shape 1 3 224 224]

- The model definition file must contain a function `get_model()` that returns an instance of the model.
- The weights file should be a state_dict or checkpoint compatible with the model.
"""
import argparse
import torch
import importlib.util
import sys


def load_model_from_file(model_def_path):
    spec = importlib.util.spec_from_file_location("model_def", model_def_path)
    model_def = importlib.util.module_from_spec(spec)
    sys.modules["model_def"] = model_def
    spec.loader.exec_module(model_def)
    if not hasattr(model_def, "get_model"):
        raise AttributeError("Model definition file must have a get_model() function.")
    return model_def.get_model()


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TorchScript format.")
    parser.add_argument('--model-def', type=str, required=True, help='Path to Python file with get_model()')
    parser.add_argument('--weights', type=str, required=True, help='Path to PyTorch weights (.pth)')
    parser.add_argument('--output', type=str, required=True, help='Output TorchScript file (.pt)')
    parser.add_argument('--input-shape', type=int, nargs='+', default=[1, 3, 224, 224], help='Input shape for tracing (default: 1 3 224 224)')
    args = parser.parse_args()

    model = load_model_from_file(args.model_def)
    state_dict = torch.load(args.weights, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    dummy_input = torch.randn(*args.input_shape)
    traced = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced, args.output)
    print(f"TorchScript model saved to {args.output}")

if __name__ == "__main__":
    main()
