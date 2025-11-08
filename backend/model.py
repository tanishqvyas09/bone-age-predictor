import torch
import torch.nn as nn
import timm
import os


def load_model(model_path, device):
    """
    Load the trained SEResNet50 model directly using timm
    """
    try:
        print(f"Loading model from: {model_path}")
        print(f"Model file exists: {os.path.exists(model_path)}")

        # Create the model directly - same way as in training
        model = timm.create_model('seresnet50', pretrained=False)
        model.reset_classifier(1)

        # Load the saved state dict
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        # Handle DataParallel wrapper if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v

        # Load weights directly into the model
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        print("✓ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


# Test model loading
if __name__ == "__main__":
    # Set device (Mac compatible)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Test loading
    model_path = 'seresnet50_final_best.pth'
    if os.path.exists(model_path):
        model = load_model(model_path, device)
        print("\nModel loaded successfully!")
        print(f"Model type: {type(model)}")

        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Test output shape: {output.shape}")
        print(f"✓ Test prediction: {output.item():.2f} months")
    else:
        print(f"Model file not found: {model_path}")
        print("Please place your .pth file in the backend folder")
        print(f"Looking for: {os.path.abspath(model_path)}")
