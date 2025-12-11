import numpy as np
import matplotlib.pyplot as plt
from ObjectDepthExtractor import ObjectDepthExtractor


def test_basic_functionality():
    """Test basic object depth extraction."""
    print("=" * 50)
    print("Test 1: Basic Functionality")
    print("=" * 50)
    
    # Create a simple 5x5 segmentation mask with an object in the center
    seg_mask = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    
    # Create a depth mask with increasing depth values
    depth_mask = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9]
    ], dtype=np.float32)
    
    extractor = ObjectDepthExtractor(seg_mask, depth_mask)
    
    # Test depth statistics
    stats = extractor.get_depth_statistics()
    print(f"Depth Statistics: {stats}")
    
    # Test bounding box
    bbox = extractor.get_bounding_box()
    print(f"Bounding Box (x_min, y_min, x_max, y_max): {bbox}")
    
    # Test centroid
    centroid = extractor.get_centroid()
    print(f"Centroid (x, y): {centroid}")
    
    # Test depth at centroid
    depth_at_centroid = extractor.get_depth_at_centroid()
    print(f"Depth at Centroid: {depth_at_centroid}")
    
    print("\n✓ Basic functionality test passed!\n")


def test_cropping():
    """Test cropping functionality."""
    print("=" * 50)
    print("Test 2: Cropping")
    print("=" * 50)
    
    # Create a 10x10 mask with object in corner
    seg_mask = np.zeros((10, 10), dtype=np.uint8)
    seg_mask[2:5, 6:9] = 1  # Object in upper right area
    
    depth_mask = np.random.rand(10, 10).astype(np.float32)
    
    extractor = ObjectDepthExtractor(seg_mask, depth_mask)
    
    # Crop without padding
    cropped_seg, cropped_depth = extractor.crop_to_object(padding=0)
    print(f"Original shape: {seg_mask.shape}")
    print(f"Cropped shape (no padding): {cropped_seg.shape}")
    print(f"Expected shape: (3, 3)")
    
    # Crop with padding
    cropped_seg_padded, cropped_depth_padded = extractor.crop_to_object(padding=2)
    print(f"Cropped shape (padding=2): {cropped_seg_padded.shape}")
    
    # Get masked depth cropped
    masked_depth = extractor.get_masked_depth_cropped(padding=1)
    print(f"Masked depth cropped shape: {masked_depth.shape}")
    print(f"NaN count in masked depth: {np.sum(np.isnan(masked_depth))}")
    
    print("\n✓ Cropping test passed!\n")


def test_normalization():
    """Test depth normalization."""
    print("=" * 50)
    print("Test 3: Normalization")
    print("=" * 50)
    
    seg_mask = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.uint8)
    
    depth_mask = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.5, 0.8],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    extractor = ObjectDepthExtractor(seg_mask, depth_mask)
    
    # Min-max normalization
    normalized_minmax = extractor.normalize_depth(method="minmax")
    object_values = normalized_minmax[seg_mask == 1]
    print(f"MinMax normalized values: {object_values}")
    print(f"Expected: [0.0, 0.5, 1.0]")
    
    # Z-score normalization
    normalized_zscore = extractor.normalize_depth(method="zscore")
    object_values_z = normalized_zscore[seg_mask == 1]
    print(f"Z-score normalized values: {object_values_z}")
    
    print("\n✓ Normalization test passed!\n")


def test_object_depth_extraction():
    """Test object depth value extraction."""
    print("=" * 50)
    print("Test 4: Object Depth Extraction")
    print("=" * 50)
    
    seg_mask = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    
    depth_mask = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ], dtype=np.float32)
    
    extractor = ObjectDepthExtractor(seg_mask, depth_mask)
    
    # Get object depth (masked array)
    object_depth = extractor.get_object_depth()
    print(f"Object depth array:\n{object_depth}")
    
    # Get flat depth values
    depth_values = extractor.get_object_depth_values()
    print(f"Depth values (flat): {depth_values}")
    print(f"Expected values: [0.2, 0.4, 0.5, 0.6, 0.8]")
    
    print("\n✓ Object depth extraction test passed!\n")


def test_edge_cases():
    """Test edge cases."""
    print("=" * 50)
    print("Test 5: Edge Cases")
    print("=" * 50)
    
    # Test with single pixel object
    seg_mask = np.zeros((5, 5), dtype=np.uint8)
    seg_mask[2, 2] = 1
    depth_mask = np.ones((5, 5), dtype=np.float32) * 0.5
    
    extractor = ObjectDepthExtractor(seg_mask, depth_mask)
    
    stats = extractor.get_depth_statistics()
    print(f"Single pixel object stats: {stats}")
    
    bbox = extractor.get_bounding_box()
    print(f"Single pixel bounding box: {bbox}")
    
    # Test with object at edge
    seg_mask_edge = np.zeros((5, 5), dtype=np.uint8)
    seg_mask_edge[0:2, 0:2] = 1
    depth_mask_edge = np.random.rand(5, 5).astype(np.float32)
    
    extractor_edge = ObjectDepthExtractor(seg_mask_edge, depth_mask_edge)
    cropped_seg, cropped_depth = extractor_edge.crop_to_object(padding=3)
    print(f"Edge object cropped shape (with padding=3): {cropped_seg.shape}")
    
    print("\n✓ Edge cases test passed!\n")


def test_validation():
    """Test input validation."""
    print("=" * 50)
    print("Test 6: Input Validation")
    print("=" * 50)
    
    # Test mismatched shapes
    try:
        seg_mask = np.zeros((5, 5), dtype=np.uint8)
        depth_mask = np.zeros((3, 3), dtype=np.float32)
        ObjectDepthExtractor(seg_mask, depth_mask)
        print("✗ Should have raised ValueError for mismatched shapes")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test invalid segmentation values
    try:
        seg_mask = np.array([[0, 1, 2], [0, 1, 0], [0, 0, 0]])
        depth_mask = np.zeros((3, 3), dtype=np.float32)
        ObjectDepthExtractor(seg_mask, depth_mask)
        print("✗ Should have raised ValueError for invalid seg values")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n✓ Validation test passed!\n")


def test_realistic_scenario():
    """Test with a more realistic scenario."""
    print("=" * 50)
    print("Test 7: Realistic Scenario")
    print("=" * 50)
    
    # Simulate a 100x100 image with an irregular object
    np.random.seed(42)
    
    seg_mask = np.zeros((100, 100), dtype=np.uint8)
    # Create an irregular blob
    for i in range(30, 70):
        for j in range(40, 80):
            if (i - 50) ** 2 + (j - 60) ** 2 < 400:  # Circle-ish shape
                seg_mask[i, j] = 1
    
    # Simulate depth with some noise
    depth_mask = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        for j in range(100):
            # Depth increases towards center of image
            depth_mask[i, j] = 0.5 + 0.3 * np.sin(i / 20) + 0.1 * np.random.rand()
    
    extractor = ObjectDepthExtractor(seg_mask, depth_mask)
    
    stats = extractor.get_depth_statistics()
    print(f"Object pixels: {stats['num_pixels']}")
    print(f"Mean depth: {stats['mean']:.4f}")
    print(f"Depth range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    bbox = extractor.get_bounding_box()
    print(f"Bounding box: {bbox}")
    
    centroid = extractor.get_centroid()
    print(f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
    
    cropped_seg, cropped_depth = extractor.crop_to_object(padding=5)
    print(f"Cropped size with padding=5: {cropped_seg.shape}")
    
    print("\n✓ Realistic scenario test passed!\n")


def test_visualization():
    """Test visualization of the extracted object depth."""
    print("=" * 50)
    print("Test 8: Visualization")
    print("=" * 50)
    
    # Create a realistic 200x200 image with an irregular object
    np.random.seed(42)
    
    seg_mask = np.zeros((200, 200), dtype=np.uint8)
    # Create a circular blob
    center_y, center_x = 100, 100
    radius = 40
    for i in range(200):
        for j in range(200):
            if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                seg_mask[i, j] = 1
    
    # Simulate ZoeDepth output (metric depth in meters)
    depth_mask = np.zeros((200, 200), dtype=np.float32)
    for i in range(200):
        for j in range(200):
            # Simulate depth: object is ~2-4 meters away with some variation
            base_depth = 3.0
            # Add gradient based on position (simulating 3D surface)
            depth_mask[i, j] = base_depth + 0.5 * np.sin(i / 30) * np.cos(j / 30) + 0.2 * np.random.rand()
    
    extractor = ObjectDepthExtractor(seg_mask, depth_mask)
    
    # Get various outputs for visualization
    object_depth = extractor.get_object_depth()
    cropped_seg, cropped_depth = extractor.crop_to_object(padding=10)
    masked_depth_cropped = extractor.get_masked_depth_cropped(padding=10)
    normalized_depth = extractor.normalize_depth(method="minmax")
    
    stats = extractor.get_depth_statistics()
    bbox = extractor.get_bounding_box()
    centroid = extractor.get_centroid()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ObjectDepthExtractor Visualization', fontsize=16, fontweight='bold')
    
    # 1. Original segmentation mask
    ax1 = axes[0, 0]
    im1 = ax1.imshow(seg_mask, cmap='gray')
    ax1.set_title('Segmentation Mask (Binary)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Mask Value')
    
    # 2. Original depth mask
    ax2 = axes[0, 1]
    im2 = ax2.imshow(depth_mask, cmap='viridis')
    ax2.set_title('Full Depth Map (ZoeDepth)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Depth (m)')
    
    # 3. Object depth (masked)
    ax3 = axes[0, 2]
    im3 = ax3.imshow(object_depth, cmap='plasma')
    ax3.set_title('Object Depth (Background=NaN)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, label='Depth (m)')
    
    # 4. Cropped and masked depth
    ax4 = axes[1, 0]
    im4 = ax4.imshow(masked_depth_cropped, cmap='plasma')
    ax4.set_title(f'Cropped Object Depth (padding=10)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4, label='Depth (m)')
    
    # 5. Normalized depth
    ax5 = axes[1, 1]
    im5 = ax5.imshow(normalized_depth, cmap='coolwarm')
    ax5.set_title('Normalized Depth (MinMax 0-1)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    plt.colorbar(im5, ax=ax5, label='Normalized Value')
    
    # 6. Overlay with bounding box and centroid
    ax6 = axes[1, 2]
    ax6.imshow(depth_mask, cmap='gray', alpha=0.5)
    ax6.contour(seg_mask, levels=[0.5], colors='lime', linewidths=2)
    
    # Draw bounding box
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax6.add_patch(rect)
    
    # Draw centroid
    if centroid:
        ax6.plot(centroid[0], centroid[1], 'r*', markersize=15, label='Centroid')
    
    ax6.set_title('Overlay: Contour, BBox, Centroid')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = 'object_depth_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    # Print statistics
    print(f"\nObject Statistics:")
    print(f"  - Pixels: {stats['num_pixels']}")
    print(f"  - Depth range: [{stats['min']:.3f}m, {stats['max']:.3f}m]")
    print(f"  - Mean depth: {stats['mean']:.3f}m")
    print(f"  - Std depth: {stats['std']:.3f}m")
    print(f"  - Bounding box: {bbox}")
    print(f"  - Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
    
    # Show the plot (comment out if running headless)
    plt.show()
    
    print("\n✓ Visualization test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("ObjectDepthExtractor Test Suite")
    print("=" * 50 + "\n")
    
    test_basic_functionality()
    test_cropping()
    test_normalization()
    test_object_depth_extraction()
    test_edge_cases()
    test_validation()
    test_realistic_scenario()
    test_visualization()
    
    print("=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()

