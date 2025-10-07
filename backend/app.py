from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as T
from PIL import Image
import io
import traceback
import cv2
import numpy as np
import timm
import os
import requests
import gdown

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'seresnet50_final_best.pth'
GDRIVE_FILE_ID = '1i4JMTJ7Rx6eJ5AH6ZahBxaWYpiDGMWjj'  # Your Google Drive file ID

# Set device (deployment compatible)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')  # Apple Silicon GPU
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


def download_model_from_gdrive():
    """Download model from Google Drive if not present"""
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already exists at {MODEL_PATH}")
        return

    print("=" * 60)
    print("📥 Downloading model from Google Drive...")
    print("=" * 60)

    try:
        # Use gdown to download from Google Drive
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✓ Model downloaded successfully!")
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("Trying alternative method...")

        # Alternative method using requests
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
            session = requests.Session()
            response = session.get(download_url, stream=True)

            # Handle large file download confirmation
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    download_url = f"https://drive.google.com/uc?export=download&confirm={value}&id={GDRIVE_FILE_ID}"
                    response = session.get(download_url, stream=True)

            # Save file
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)

            print("✓ Model downloaded successfully (alternative method)!")
        except Exception as e2:
            print(f"✗ Failed to download model: {e2}")
            raise


def load_model(model_path, device):
    """Load the trained SEResNet50 model"""
    print(f"Loading model from: {model_path}")

    # Create the model - same way as in training
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

    # Load weights
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print("✓ Model loaded successfully!")
    return model


# Download and load model at startup
print("=" * 60)
print("🚀 Starting Bone Age Prediction Server")
print("=" * 60)

# Download model if needed
download_model_from_gdrive()

# Load model
model = load_model(MODEL_PATH, device)
print("=" * 60)

# Image preprocessing - EXACTLY as in your training code
transform = T.Compose([
    T.Resize((512, 512)),  # Same as training
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])


def preprocess_image(image):
    """
    Preprocess image with CLAHE - exactly as in training
    """
    # Convert PIL to numpy array
    img = np.array(image)

    # Convert to grayscale if not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Convert back to RGB (3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Convert back to PIL Image
    img = Image.fromarray(img)

    return img


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Bone Age Prediction API',
        'status': 'running',
        'model': 'SEResNet50',
        'device': str(device),
        'version': '1.0.0'
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    })


@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    try:
        # Read image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        print(f"Original image: {img.size}, mode: {img.mode}")

        # Apply CLAHE preprocessing (same as training)
        img = preprocess_image(img)
        print("✓ CLAHE preprocessing applied")

        # Apply transforms
        img_tensor = transform(img).unsqueeze(0).to(device)
        print(f"✓ Tensor shape: {img_tensor.shape}")

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            predicted_age_months = output.squeeze().item()

        print(f"✓ Prediction: {predicted_age_months:.2f} months")

        # Format response
        return jsonify({
            'success': True,
            'predicted_age_months': round(predicted_age_months, 2),
            'predicted_age_years': round(predicted_age_months / 12, 2),
            'device_used': str(device)
        })

    except Exception as e:
        print(f"✗ Error during prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify server is running"""
    return jsonify({
        'message': 'Server is running!',
        'model_loaded': model is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    # Get port from environment variable (for deployment) or use 5001 (for local)
    port = int(os.environ.get('PORT', 5001))

    print("=" * 60)
    print("🦴 Bone Age Prediction Server (SEResNet50)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: SEResNet50")
    print(f"Server: http://0.0.0.0:{port}")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /          - API info")
    print("  GET  /health    - Health check")
    print("  GET  /test      - Test endpoint")
    print("  POST /predict   - Predict bone age")
    print("=" * 60)

    # Use 0.0.0.0 to allow external connections
    app.run(debug=False, host='0.0.0.0', port=port)
