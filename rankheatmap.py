import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTModel
from PIL import Image, ImageOps
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

# ------------------------------
# HEATMAP GENERATION
# ------------------------------
def generate_best_heatmap(image_path, threshold=0.1):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    processor = ViTImageProcessor.from_pretrained("facebook/dino-vits8", size=480)
    model = ViTModel.from_pretrained(
        "facebook/dino-vits8", add_pooling_layer=False, attn_implementation="eager"
    )

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)

    attentions = outputs.attentions[-1]
    nh = attentions.shape[1]
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    w_featmap = pixel_values.shape[-2] // model.config.patch_size
    h_featmap = pixel_values.shape[-1] // model.config.patch_size

    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)

    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest"
    )[0].cpu().detach().numpy()

    mean_scores = attentions.mean(axis=(1, 2))
    best_head = np.argmax(mean_scores)
    best_map = attentions[best_head]

    return best_map, image

# ------------------------------
# SUBJECT CLUSTERING
# ------------------------------
def cluster_nearby_regions(subjects, distance_threshold=100, min_overlap_ratio=0.1):
    """
    Cluster nearby subjects that likely belong to the same person
    
    Args:
        subjects: List of detected subject dictionaries
        distance_threshold: Maximum distance between centers to consider clustering
        min_overlap_ratio: Minimum overlap ratio to force clustering
    """
    if len(subjects) <= 1:
        return subjects
    
    # Extract centers and bounding boxes
    centers = []
    bboxes = []
    for subj in subjects:
        x, y, w, h = subj["bbox"]
        centers.append([x + w/2, y + h/2])
        bboxes.append([x, y, x+w, y+h])
    
    centers = np.array(centers)
    bboxes = np.array(bboxes)
    
    # Calculate distance matrix
    distances = cdist(centers, centers)
    
    # Calculate overlap matrix
    overlap_matrix = np.zeros((len(subjects), len(subjects)))
    for i in range(len(subjects)):
        for j in range(i+1, len(subjects)):
            x1_min, y1_min, x1_max, y1_max = bboxes[i]
            x2_min, y2_min, x2_max, y2_max = bboxes[j]
            
            # Calculate overlap
            overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            overlap_area = overlap_x * overlap_y
            
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            
            # Overlap ratio relative to smaller box
            overlap_ratio = overlap_area / min(area1, area2) if min(area1, area2) > 0 else 0
            overlap_matrix[i, j] = overlap_matrix[j, i] = overlap_ratio
    
    # Create adjacency matrix for clustering
    adjacency = np.zeros((len(subjects), len(subjects)))
    
    for i in range(len(subjects)):
        for j in range(i+1, len(subjects)):
            # Connect if close distance OR significant overlap
            if (distances[i, j] < distance_threshold or 
                overlap_matrix[i, j] > min_overlap_ratio):
                adjacency[i, j] = adjacency[j, i] = 1
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    def dfs(node, cluster):
        visited.add(node)
        cluster.append(node)
        for neighbor in range(len(subjects)):
            if adjacency[node, neighbor] == 1 and neighbor not in visited:
                dfs(neighbor, cluster)
    
    for i in range(len(subjects)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)
    
    return clusters

def merge_clustered_subjects(subjects, clusters, heatmap):
    """
    Merge subjects in each cluster into single bounding boxes
    """
    merged_subjects = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            # Single subject - keep as is
            merged_subjects.append(subjects[cluster[0]])
        else:
            # Multiple subjects - merge them
            cluster_subjects = [subjects[i] for i in cluster]
            
            # Find bounding box that encompasses all subjects in cluster
            min_x = min(subj["bbox"][0] for subj in cluster_subjects)
            min_y = min(subj["bbox"][1] for subj in cluster_subjects)
            max_x = max(subj["bbox"][0] + subj["bbox"][2] for subj in cluster_subjects)
            max_y = max(subj["bbox"][1] + subj["bbox"][3] for subj in cluster_subjects)
            
            merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            # Calculate combined metrics
            total_area = sum(subj["area"] for subj in cluster_subjects)
            weighted_mean_score = sum(subj["mean_score"] * subj["area"] for subj in cluster_subjects) / total_area
            max_ranking_score = max(subj["ranking_score"] for subj in cluster_subjects)
            
            merged_subjects.append({
                "bbox": merged_bbox,
                "area": int(total_area),
                "mean_score": float(weighted_mean_score),
                "ranking_score": float(max_ranking_score),
                "cluster_size": len(cluster)
            })
    
    # Sort merged subjects by ranking score
    merged_subjects = sorted(merged_subjects, key=lambda s: s["ranking_score"], reverse=True)
    
    return merged_subjects

# ------------------------------
# ENHANCED MULTI-SUBJECT DETECTION
# ------------------------------
def detect_and_cluster_subjects(heatmap, threshold_percentile=85, min_area=500, 
                               top_k=10, distance_threshold=120, final_top_k=5):
    """
    Detect subjects and cluster nearby regions
    
    Args:
        top_k: Initial number of subjects to detect (before clustering)
        distance_threshold: Maximum distance between centers for clustering
        final_top_k: Final number of subjects after clustering
    """
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    threshold_value = np.percentile(heatmap_norm, threshold_percentile)
    binary_mask = (heatmap_norm >= threshold_value).astype(np.uint8) * 255

    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    subjects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros_like(binary_mask)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        mean_score = np.mean(heatmap_norm[mask == 1])
        ranking_score = mean_score * area
        
        subjects.append({
            "bbox": (x, y, w, h),
            "area": int(area),
            "mean_score": float(mean_score),
            "ranking_score": float(ranking_score)
        })

    # Sort by importance and take top candidates
    subjects = sorted(subjects, key=lambda s: s["ranking_score"], reverse=True)[:top_k]
    
    # Cluster nearby subjects
    clusters = cluster_nearby_regions(subjects, distance_threshold)
    
    # Merge clustered subjects
    merged_subjects = merge_clustered_subjects(subjects, clusters, heatmap)
    
    return merged_subjects[:final_top_k]

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def plot_original_and_clustered_heatmap(image_path, output_dir="outputs",
                                       threshold=0.1, subject_threshold=85,
                                       min_area=500, distance_threshold=120,
                                       initial_top_k=10, final_top_k=3):
    os.makedirs(output_dir, exist_ok=True)
    best_map, image = generate_best_heatmap(image_path, threshold)
    subjects = detect_and_cluster_subjects(
        best_map, subject_threshold, min_area, 
        initial_top_k, distance_threshold, final_top_k
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Heatmap with clustered subjects
    axes[1].imshow(best_map, cmap="jet")
    axes[1].set_title(f"Clustered Subject Detection ({len(subjects)} subjects)")
    axes[1].axis("off")

    # Draw bounding boxes with different colors
    colors = ['red', 'yellow', 'cyan', 'magenta', 'lime', 'orange', 'pink']
    for i, subj in enumerate(subjects):
        x, y, w, h = subj["bbox"]
        color = colors[i % len(colors)]
        rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=3)
        axes[1].add_patch(rect)
        
        # Show subject number and cluster info
        label = f"{i+1}"
        if "cluster_size" in subj and subj["cluster_size"] > 1:
            label += f" ({subj['cluster_size']} merged)"
            
        axes[1].text(x, y - 5, label, color="white",
                     fontsize=10, fontweight="bold",
                     bbox=dict(facecolor=color, alpha=0.8))

    out_path = os.path.join(output_dir, "clustered_subject_detection.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    # Print analysis
    print(f"✅ Detected {len(subjects)} clustered subjects:")
    for i, subj in enumerate(subjects):
        cluster_info = f" (merged {subj['cluster_size']} regions)" if "cluster_size" in subj else ""
        print(f"  Subject {i+1}: area={subj['area']}, score={subj['mean_score']:.3f}{cluster_info}")
    
    print(f"✅ Saved result: {out_path}")
    return subjects

# ------------------------------
# RUN WITH CLUSTERING
# ------------------------------
if __name__ == "__main__":
    image_path = r"C:\Users\HP\OneDrive\Desktop\photo\joly\SBCL5999.JPG"
    
    # Detect with clustering
    subjects = plot_original_and_clustered_heatmap(
        image_path,
        threshold=0.1,
        subject_threshold=85,
        min_area=500,
        distance_threshold=120,  # Adjust this to control clustering sensitivity
        initial_top_k=10,       # Detect more regions initially
        final_top_k=3           # Return fewer final subjects
    )