import coremltools as ct
import numpy as np

# Path to your CoreML model
MODEL_PATH = '/Users/hc/Documents/coreml/best_model_B_tuned.mlpackage'

# Load the CoreML model
model = ct.models.MLModel(MODEL_PATH)

# Prepare a dummy input (replace with real preprocessed image for real inference)
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
input_dict = {"image": input_data}

# Run inference
output = model.predict(input_dict)
print("Model output:", output)
