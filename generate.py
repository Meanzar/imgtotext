from PIL import Image
import torch
from model import model, feature_extractor, tokenizer, device

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

if __name__ == "__main__":
    # Example usage
    image_paths = ["images.jpeg", "lionn.jpeg"]
    captions = predict_step(image_paths)
    for img, caption in zip(image_paths, captions):
        print(f"Image: {img} - Caption: {caption}")