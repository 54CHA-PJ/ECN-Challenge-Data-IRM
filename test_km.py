import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, measure, filters
from skimage.filters import gaussian
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import binary_fill_holes, median_filter
from skimage.measure import label as sk_label, regionprops
from skimage.segmentation import find_boundaries
import networkx as nx


def process_single_image(input_image):
    print("Starting pipeline...")
    h, w = input_image.shape
    start_time = time.time()

    print("Preprocessing image...")
    image_float = img_as_float(input_image)
    smoothed = gaussian(image_float, sigma=1)
    smoothed_uint8 = (smoothed * 255).astype("uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(smoothed_uint8)
    normalized = equalized / 255.0
    denoised = median_filter(normalized, size=5)
    resized = resize(denoised, (128, 128), anti_aliasing=True)

    print("Extracting ROI...")
    threshold = filters.threshold_otsu(resized)
    binary_mask = resized > threshold
    labeled_mask = sk_label(binary_mask)
    regions = measure.regionprops(labeled_mask)
    largest_region = max(regions, key=lambda r: r.area)
    continuous_mask = labeled_mask == largest_region.label
    roi_mask = binary_fill_holes(continuous_mask)

    print("Quantizing image...")
    quantized_image = np.floor(resized * 8).astype(np.uint8)
    quantized_image[quantized_image == 8] = 7

    print("Preparing data for clustering...")
    X, Y = np.meshgrid(np.arange(128), np.arange(128))
    features = np.stack((resized.flatten(), X.flatten(), Y.flatten()), axis=-1)
    features_scaled = StandardScaler().fit_transform(features)

    print("Running KMeans clustering...")
    kmeans = KMeans(n_clusters=80, n_init=10, tol=1e-4, algorithm="elkan")
    kmeans.fit(features_scaled)
    cluster_labels = kmeans.labels_.reshape(128, 128)

    print("Refining clusters...")
    def refine_kmeans_with_roi(kmeans_labels, roi_mask):
        refined = np.copy(kmeans_labels)
        refined[~roi_mask] = 0
        unique_labels, new_labels = np.unique(refined, return_inverse=True)
        refined = new_labels.reshape(refined.shape) + 1
        refined[~roi_mask] = 0
        return refined

    refined_clusters = refine_kmeans_with_roi(cluster_labels, roi_mask)

    print("Removing small clusters...")
    def remove_small_clusters(segmented_image, min_size=20):
        refined_image = np.copy(segmented_image)
        for label_value in np.unique(segmented_image):
            if label_value == 0:
                continue
            cluster_mask = (segmented_image == label_value)
            if np.sum(cluster_mask) < min_size:
                refined_image[cluster_mask] = 0
        return sk_label(refined_image, background=0)

    cleaned_clusters = remove_small_clusters(refined_clusters)

    print("Merging clusters based on quantized values...")
    def calculate_cluster_modes(segmented_image, quantized_image):
        cluster_modes = {}
        for label_value in np.unique(segmented_image):
            if label_value == 0:
                continue
            cluster_mask = segmented_image == label_value
            values, counts = np.unique(quantized_image[cluster_mask], return_counts=True)
            cluster_modes[label_value] = values[np.argmax(counts)]
        return cluster_modes

    cluster_modes = calculate_cluster_modes(cleaned_clusters, quantized_image)

    def merge_clusters(segmented_image, cluster_modes):
        G = nx.Graph()
        labels = np.unique(segmented_image)
        G.add_nodes_from(labels)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dy, dx in shifts:
            shifted = np.roll(segmented_image, shift=(dy, dx), axis=(0, 1))
            mask = (segmented_image != shifted) & (segmented_image > 0) & (shifted > 0)
            for a, b in zip(segmented_image[mask], shifted[mask]):
                if cluster_modes.get(a) == cluster_modes.get(b):
                    G.add_edge(a, b)

        label_mapping = {}
        for new_label, cluster in enumerate(nx.connected_components(G)):
            for label in cluster:
                label_mapping[label] = new_label

        merged_image = np.copy(segmented_image)
        for old_label, new_label in label_mapping.items():
            merged_image[segmented_image == old_label] = new_label

        return merged_image

    merged_clusters = merge_clusters(cleaned_clusters, cluster_modes)

    print("Removing clusters touching background and noise...")
    def remove_noise_and_background(segmented_image, min_size=10):
        cleaned_image = np.copy(segmented_image)
        labeled_mask = sk_label(segmented_image)
        for region in measure.regionprops(labeled_mask):
            if region.area < min_size:
                cleaned_image[labeled_mask == region.label] = 0
        return sk_label(cleaned_image, background=0)

    final_cleaned_image = remove_noise_and_background(merged_clusters)

    print("Resizing to original dimensions...")
    final_image = resize(final_cleaned_image, (h, w), order=0, preserve_range=True).astype(np.uint8)

    print("Plotting results...")
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(resized, cmap="gray")
    axs[0].set_title("Preprocessed Image")
    axs[0].axis("off")

    axs[1].imshow(refined_clusters, cmap="nipy_spectral")
    axs[1].set_title("Refined Clusters")
    axs[1].axis("off")

    axs[2].imshow(merged_clusters, cmap="nipy_spectral")
    axs[2].set_title("Merged Clusters")
    axs[2].axis("off")

    axs[3].imshow(final_image, cmap="nipy_spectral")
    axs[3].set_title("Final Image")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Pipeline completed in {time.time() - start_time:.2f} seconds")
    return final_image


if __name__ == "__main__":
    import cv2
    from pathlib import Path
    import os
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'
    img_path = Path(r"C:\Users\sacha\Desktop\PROJ_DATA\Medical\data\images_hTE3Lse\images\6.png")
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    process_single_image(image)
