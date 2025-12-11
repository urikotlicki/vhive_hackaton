import numpy as np
import matplotlib.pyplot as plt
from ObjectDepthExtractor import ObjectDepthExtractor


def load_masks(segmentation_path: str, depth_path: str) -> tuple:
    """
    Load segmentation and depth masks from npy files.
    
    Args:
        segmentation_path: Path to the segmentation mask npy file (binary 0/1 values).
        depth_path: Path to the depth mask npy file (from ZoeDepth model).
    
    Returns:
        Tuple of (segmentation_mask, depth_mask) numpy arrays.
    """
    segmentation_mask = np.load(segmentation_path)
    depth_mask = np.load(depth_path)
    
    print(f"Loaded segmentation mask: {segmentation_path}")
    print(f"  Shape: {segmentation_mask.shape}, dtype: {segmentation_mask.dtype}")
    
    print(f"Loaded depth mask: {depth_path}")
    print(f"  Shape: {depth_mask.shape}, dtype: {depth_mask.dtype}")
    
    return segmentation_mask, depth_mask


def visualize_results(extractor: ObjectDepthExtractor, output_path: str = None):
    """
    Visualize the extraction results.
    
    Args:
        extractor: ObjectDepthExtractor instance.
        output_path: Optional path to save the visualization.
    """
    object_depth = extractor.get_object_depth()
    masked_depth_cropped = extractor.get_masked_depth_cropped(padding=10)
    normalized_depth = extractor.normalize_depth(method="minmax")
    
    stats = extractor.get_depth_statistics()
    bbox = extractor.get_bounding_box()
    centroid = extractor.get_centroid()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ObjectDepthExtractor Results', fontsize=16, fontweight='bold')
    
    # 1. Segmentation mask
    ax1 = axes[0, 0]
    im1 = ax1.imshow(extractor.segmentation_mask, cmap='gray')
    ax1.set_title('Segmentation Mask')
    plt.colorbar(im1, ax=ax1, label='Mask Value')
    
    # 2. Full depth mask
    ax2 = axes[0, 1]
    im2 = ax2.imshow(extractor.depth_mask, cmap='viridis')
    ax2.set_title('Full Depth Map (ZoeDepth)')
    plt.colorbar(im2, ax=ax2, label='Depth (m)')
    
    # 3. Object depth
    ax3 = axes[0, 2]
    im3 = ax3.imshow(object_depth, cmap='plasma')
    ax3.set_title('Object Depth')
    plt.colorbar(im3, ax=ax3, label='Depth (m)')
    
    # 4. Cropped depth
    ax4 = axes[1, 0]
    im4 = ax4.imshow(masked_depth_cropped, cmap='plasma')
    ax4.set_title('Cropped Object Depth')
    plt.colorbar(im4, ax=ax4, label='Depth (m)')
    
    # 5. Normalized depth
    ax5 = axes[1, 1]
    im5 = ax5.imshow(normalized_depth, cmap='coolwarm')
    ax5.set_title('Normalized Depth (0-1)')
    plt.colorbar(im5, ax=ax5, label='Normalized')
    
    # 6. Overlay
    ax6 = axes[1, 2]
    ax6.imshow(extractor.depth_mask, cmap='gray', alpha=0.5)
    ax6.contour(extractor.segmentation_mask, levels=[0.5], colors='lime', linewidths=2)
    
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax6.add_patch(rect)
    
    if centroid:
        ax6.plot(centroid[0], centroid[1], 'r*', markersize=15, label='Centroid')
        ax6.legend(loc='upper right')
    
    ax6.set_title('Overlay: Contour, BBox, Centroid')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


def main(segmentation_npy_path: str, depth_npy_path: str):
    """
    Main function to process segmentation and depth masks.
    
    Args:
        segmentation_npy_path: Path to segmentation mask npy file (binary 0/1 values).
        depth_npy_path: Path to depth mask npy file (from ZoeDepth model).
    """
    # Load masks
    print("=" * 50)
    print("Loading input files...")
    print("=" * 50)
    segmentation_mask, depth_mask = load_masks(segmentation_npy_path, depth_npy_path)
    
    # Create extractor
    print("\n" + "=" * 50)
    print("Processing...")
    print("=" * 50)
    extractor = ObjectDepthExtractor(segmentation_mask, depth_mask)
    
    # Get statistics
    stats = extractor.get_depth_statistics()
    bbox = extractor.get_bounding_box()
    centroid = extractor.get_centroid()
    
    print(f"\nObject Statistics:")
    print(f"  - Number of pixels: {stats['num_pixels']}")
    print(f"  - Depth range: [{stats['min']:.3f}m, {stats['max']:.3f}m]")
    print(f"  - Mean depth: {stats['mean']:.3f}m")
    print(f"  - Median depth: {stats['median']:.3f}m")
    print(f"  - Std depth: {stats['std']:.3f}m")
    print(f"  - Bounding box (x_min, y_min, x_max, y_max): {bbox}")
    if centroid:
        print(f"  - Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
    
    # Visualize
    print("\n" + "=" * 50)
    print("Generating visualization...")
    print("=" * 50)
    visualize_results(extractor)
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    # Set paths to your npy files here
    SEGMENTATION_NPY_PATH = "path/to/segmentation_mask.npy"
    DEPTH_NPY_PATH = "path/to/depth_mask.npy"
    
    main(SEGMENTATION_NPY_PATH, DEPTH_NPY_PATH)
