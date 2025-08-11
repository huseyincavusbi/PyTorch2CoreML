"""
check_torchscript.py

Check if a given PyTorch model file is a TorchScript model.
Usage: python check_torchscript.py --model <path_to_model_file>
"""
import argparse
import torch

def is_torchscript_model(model_path):
    try:
        model = torch.jit.load(model_path, map_location='cpu')
        # TorchScript models are instances of ScriptModule or RecursiveScriptModule
        if isinstance(model, torch.jit.ScriptModule) or isinstance(model, torch.jit.RecursiveScriptModule):
            return True
        # For PyTorch >=1.9, ScriptFunction is also possible
        if hasattr(torch.jit, 'ScriptFunction') and isinstance(model, torch.jit.ScriptFunction):
            return True
        return False
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Check if a PyTorch model file is TorchScript.")
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pt or .pth)')
    args = parser.parse_args()
    if is_torchscript_model(args.model):
        print(f"{args.model} is a TorchScript model.")
    else:
        print(f"{args.model} is NOT a TorchScript model.")

if __name__ == "__main__":
    main()
