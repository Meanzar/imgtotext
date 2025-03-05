from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Load the model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Export the necessary components
__all__ = ['model', 'feature_extractor', 'tokenizer', 'device']