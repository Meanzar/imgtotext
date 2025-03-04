import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from model import model, feature_extractor, tokenizer, device
from datasets import load_dataset
from PIL import Image

# Hyperparameters
epochs = 10
learning_rate = 0.001
max_data_length = 1000  # Set the maximum length of the data

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load COCO dataset
coco_dataset = load_dataset("jxie/coco_captions", split='train')

# Define a custom dataset class to apply transformations
class CocoCaptionsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self.dataset[idx]['caption']
        return image, caption

# Create dataset and dataloader
coco_dataset = CocoCaptionsDataset(coco_dataset, transform=transform)
subset_indices = list(range(min(max_data_length, len(coco_dataset))))
coco_subset = Subset(coco_dataset, subset_indices)
data_loader = DataLoader(coco_subset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    model.train()
    for epoch in range(epochs):
        for i, (images, captions) in enumerate(data_loader):
            images = images.to(device)
            tokenized_captions = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized_captions.input_ids.to(device)
            

            # Forward pass
            outputs = model(pixel_values=images, labels=input_ids)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            print(f'Epoch {epoch}, step {i}, loss = {loss.item()}')

if __name__ == "__main__":
    train()