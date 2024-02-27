from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import torch
# This cell ensures future use (even after PC rr) won't require downloading the above packages (600+MB)
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def label_area():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Path to your satellite image
    image_path = r'../data/images/satellite_image.png'
    
    # Load the image
    image = Image.open(image_path)
    
    # Define your text queries
    text_queries = ["rural","suburban","urban"]  # Add or modify as needed
    
    
    # Process the inputs
    inputs = processor(text=text_queries, images=image, return_tensors="pt", padding=True)
    
    # Get model outputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    argmax_index = torch.argmax(probs)
    return text_queries[argmax_index.item()]

