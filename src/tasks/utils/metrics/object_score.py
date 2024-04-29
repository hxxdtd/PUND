# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
import torch

object_label = {
    'jeep': 609,
    'english springer': 217,
}

# Initialize model with the best available weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).cuda()
model.eval()

# Initialize the inference transforms
preprocess = weights.transforms()

def calculate_object_score(image_path, object):
    with torch.no_grad():
        image = read_image(image_path)
        # Apply inference preprocessing transforms
        batch = preprocess(image).unsqueeze(0).cuda()

        # Use the model and print the predicted category
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = object_label[object]
        score = float(prediction[class_id].item())
    return score