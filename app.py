from flask import Flask, request, render_template, send_file
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from model_definition import ColorAutoEncoder  # Replace with your model class

app = Flask(__name__)

# Load the model
model = ColorAutoEncoder()
model.load_state_dict(torch.load('colorization_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

inverse_transform = transforms.ToPILImage()

@app.route('/')
def upload_page():
    return render_template('upload.html')  # Replace with your HTML page name

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # Open the image and preprocess
    img = Image.open(file)
    gray_img = transform(img).unsqueeze(0)  # Add batch dimension

    # Predict colorized image
    with torch.no_grad():
        output = model(gray_img)

    # Convert output tensor to image
    color_img = inverse_transform(output.squeeze(0))

    # Save or send the image
    img_io = io.BytesIO()
    color_img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
