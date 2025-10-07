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
import gdown
import gc

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'seresnet50_final_best.pth'
GDRIVE_FILE_ID = '1i4JMTJ7Rx6eJ5AH6ZahBxaWYpiDGMWjj'

# Force CPU and optimize for low memory
device = torch.device('cpu')
torch.set_num_threads(1)  # Reduce thread overhead
print(f"Using device: {device}")

# Global model variable (lazy loading)
model = None


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
        raise


def load_model_optimized(model_path, device):
    """Load model with aggressive memory optimization"""
    print(f"Loading model from: {model_path}")
    
    # Clear memory before loading
    gc.collect()
    
    # Create model
    model = timm.create_model('seresnet50', pretrained=False)
    model.reset_classifier(1)
    
    # Load state dict with memory mapping
    print("Loading weights...")
    state_dict = torch.load(
        model_path, 
        map_location=device,
        weights_only=False
    )
    
    # Handle DataParallel wrapper
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        new_state_dict[key] = v
    
    # Load weights and clean up immediately
    model.load_state_dict(new_state_dict)
    del state_dict, new_state_dict
    gc.collect()
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Freeze all parameters to save memory
    for param in model.parameters():
        param.requires_grad = False
    
    print("✓ Model loaded successfully!")
    return model


def get_model():
    """Lazy load model on first request"""
    global model
    if model is None:
        print("First request - initializing model...")
        download_model_from_gdrive()
        model = load_model_optimized(MODEL_PATH, device)
        gc.collect()
    return model


# Image preprocessing transforms
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess_image(image):
    """Preprocess image with CLAHE"""
    # Convert PIL to numpy
    img = np.array(image)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Convert back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(img)


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
        # Get model (lazy load on first request)
        current_model = get_model()
        
        # Read image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        print(f"Original image: {img.size}, mode: {img.mode}")

        # Apply CLAHE preprocessing
        img = preprocess_image(img)
        print("✓ CLAHE preprocessing applied")

        # Apply transforms
        img_tensor = transform(img).unsqueeze(0).to(device)
        print(f"✓ Tensor shape: {img_tensor.shape}")

        # Predict
        with torch.no_grad():
            output = current_model(img_tensor)
            predicted_age_months = output.squeeze().item()

        # Clean up tensors immediately
        del img_tensor, output
        gc.collect()

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
    print(f"Model: SEResNet50 (Lazy Loading)")
    print(f"Memory Optimization: Enabled")
    print(f"Server: http://0.0.0.0:{port}")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /          - API info")
    print("  GET  /health    - Health check")
    print("  GET  /test      - Test endpoint")
    print("  POST /predict   - Predict bone age")
    print("=" * 60)
    print("\n⚠️  Note: Model loads on first prediction request")
    print("=" * 60)

    # Use 0.0.0.0 to allow external connections
    app.run(debug=False, host='0.0.0.0', port=port, threaded=False)
