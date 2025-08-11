import torch
import timm
import torch.nn as nn

# Model definition for export and conversion
# This must match the architecture and parameters used in training

def get_model():
    FINAL_MODEL_NAME = 'seresnext101_32x8d'
    NUM_CLASSES = 7
    DROP_RATE = 0.21224189367269414
    # pretrained=True is safe for export, but weights will be overwritten by checkpoint
    model = timm.create_model(FINAL_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_rate=DROP_RATE)
    return model
