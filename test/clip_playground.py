import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

import pdb; pdb.set_trace()


image = preprocess(Image.open("apple.png")).unsqueeze(0).to(device)
text = clip.tokenize(["picking up an apple", "picking up a banana", "picking up a plum", "picking up an orange"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 