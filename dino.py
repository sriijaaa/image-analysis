
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTModel
from PIL import Image, ImageOps


def generate_best_heatmap(image_path, output_dir="outputs", threshold=0.1):
    os.makedirs(output_dir, exist_ok=True)

    # Load image with correct orientation
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # fix rotation

    # Load processor + model
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vits8", size=480)
    model = ViTModel.from_pretrained(
        "facebook/dino-vits8", add_pooling_layer=False, attn_implementation="eager"
    )

    # Extract pixel values
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    print("Pixel values shape:", pixel_values.shape)

    # Forward pass
    outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)
    attentions = outputs.attentions[-1]  # last layer
    nh = attentions.shape[1]  # number of heads

    # Only patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # Feature map size
    w_featmap = pixel_values.shape[-2] // model.config.patch_size
    h_featmap = pixel_values.shape[-1] // model.config.patch_size

    # Thresholding
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    # Interpolate to original image size
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest"
    )[0].cpu().detach().numpy()

    # Pick the "best" head = highest mean activation
    mean_scores = attentions.mean(axis=(1, 2))
    best_head = np.argmax(mean_scores)
    best_map = attentions[best_head]

    # --- Save original + heatmap side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: input image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Right: best heatmap
    axes[1].imshow(best_map, cmap="jet")
    axes[1].set_title("Best Heatmap")
    axes[1].axis("off")

    combined_path = os.path.join(output_dir, "original_and_heatmap.png")
    plt.savefig(combined_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"✅ Combined result saved as {combined_path}")


if __name__ == "__main__":
    # 👇 Replace this with your image path
    image_path = r"C:\Users\HP\OneDrive\Desktop\photo\pics\DSC07951.JPG"
    generate_best_heatmap(image_path, threshold=0.1)  