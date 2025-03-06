import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from datasets import load_dataset
from PIL import Image

# Hyperparameters
epochs = 10
learning_rate = 0.001
# Limit the number of data samples to use for training
max_data_length = 4096

# Load the pretrained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freeze the vision encoder layers (ViT encoder)
for param in model.encoder.parameters():
    param.requires_grad = False


# Load COCO dataset
coco_dataset = load_dataset("jxie/coco_captions", split='train')

# Custom dataset class
class CocoCaptionsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_extractor):
        self.dataset = dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Preprocess image
        processed = self.feature_extractor(images=image, return_tensors="pt")['pixel_values']
        processed = processed.squeeze(0)  # Remove batch dimension

        caption = self.dataset[idx]['caption']
        return processed, caption

# Create dataset and dataloader
coco_dataset = CocoCaptionsDataset(coco_dataset, feature_extractor)
subset_indices = list(range(min(max_data_length, len(coco_dataset))))
coco_subset = Subset(coco_dataset, subset_indices)
data_loader = DataLoader(coco_subset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    model.train()
    for epoch in range(epochs):
        for step, (images, captions) in enumerate(data_loader):
            images = images.to(device)  

            # Convert captions to input_ids
            tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = tokenized.input_ids.to(device)
            
            optimizer.zero_grad()  
            outputs = model(pixel_values=images, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, step {step+1}, loss = {loss.item()}')

    # Save the trained model state
    torch.save(model.state_dict(), "imgtotext_transformer.pth")
    print("Model saved.")    

if __name__ == "__main__":
    train()
