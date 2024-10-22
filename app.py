import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Step 1: Load the Trained Model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 11)  # 11 weather classes
model.load_state_dict(torch.load('resnet50_20epochs.pth',map_location=torch.device('cpu')))
model.eval()

# Step 2: Define the Prediction Function
def predict(image):
    # Transform the image to match the model's expected input
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)
    
    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    class_names = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'sunny', 'rime', 'sandstorm', 'snow']
    return class_names[predicted_class.item()]

# Step 3: Create a Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Weather Classification Model",
    description="Upload an image to classify the type of weather."
)

# Step 4: Launch the Interface
interface.launch()