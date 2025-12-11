import numpy as np
from typing import Optional, Tuple


class ObjectDepthExtractor:
    """
    Extracts depth information for a specific object using segmentation and depth masks.
    
    The segmentation mask contains binary values (0/1) where 1 indicates the object region.
    The depth mask is obtained from ZoeDepth model (external, not implemented here).
    """
    
    def __init__(self, segmentation_mask: np.ndarray, depth_mask: np.ndarray):
        """
        Initialize the ObjectDepthExtractor.
        
        Args:
            segmentation_mask: Binary mask (H, W) with 0/1 values. 
                               1 indicates object pixels, 0 indicates background.
            depth_mask: Depth map (H, W) from ZoeDepth model.
                        Values represent metric depth in meters.
        
        Raises:
            ValueError: If masks have different shapes or invalid values.
        """
        self._validate_inputs(segmentation_mask, depth_mask)
        
        self.segmentation_mask = segmentation_mask.astype(np.uint8)
        self.depth_mask = depth_mask.astype(np.float32)
        self.height, self.width = segmentation_mask.shape
        
    def _validate_inputs(self, segmentation_mask: np.ndarray, depth_mask: np.ndarray) -> None:
        """Validate input masks."""
        if segmentation_mask.shape != depth_mask.shape:
            raise ValueError(
                f"Mask shapes must match. Got segmentation: {segmentation_mask.shape}, "
                f"depth: {depth_mask.shape}"
            )
        
        if segmentation_mask.ndim != 2:
            raise ValueError(f"Masks must be 2D. Got {segmentation_mask.ndim} dimensions.")
        
        unique_values = np.unique(segmentation_mask)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                f"Segmentation mask must contain only 0 and 1 values. "
                f"Found: {unique_values}"
            )
    
    def get_object_depth(self) -> np.ndarray:
        """
        Extract depth values only for the segmented object.
        
        Returns:
            Masked depth array where background pixels are set to NaN.
        """
        object_depth = self.depth_mask.copy()
        object_depth[self.segmentation_mask == 0] = np.nan
        return object_depth
    
    def get_object_depth_values(self) -> np.ndarray:
        """
        Get flat array of depth values for object pixels only.
        
        Returns:
            1D array of depth values for the object region.
        """
        return self.depth_mask[self.segmentation_mask == 1]
    
    def get_depth_statistics(self) -> dict:
        """
        Calculate depth statistics for the segmented object.
        
        Returns:
            Dictionary containing min, max, mean, median, and std of depth values.
        """
        depth_values = self.get_object_depth_values()
        
        if len(depth_values) == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "num_pixels": 0
            }
        
        return {
            "min": float(np.min(depth_values)),
            "max": float(np.max(depth_values)),
            "mean": float(np.mean(depth_values)),
            "median": float(np.median(depth_values)),
            "std": float(np.std(depth_values)),
            "num_pixels": len(depth_values)
        }
    
    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of the segmented object.
        
        Returns:
            Tuple (x_min, y_min, x_max, y_max) or None if no object pixels exist.
        """
        rows = np.any(self.segmentation_mask, axis=1)
        cols = np.any(self.segmentation_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    def crop_to_object(self, padding: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop both masks to the object's bounding box with optional padding.
        
        Args:
            padding: Number of pixels to pad around the bounding box.
        
        Returns:
            Tuple of (cropped_segmentation, cropped_depth) arrays.
        
        Raises:
            ValueError: If no object pixels exist in the segmentation mask.
        """
        bbox = self.get_bounding_box()
        if bbox is None:
            raise ValueError("No object pixels found in segmentation mask.")
        
        x_min, y_min, x_max, y_max = bbox
        
        # Apply padding with bounds checking
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(self.width - 1, x_max + padding)
        y_max = min(self.height - 1, y_max + padding)
        
        cropped_seg = self.segmentation_mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_depth = self.depth_mask[y_min:y_max + 1, x_min:x_max + 1]
        
        return cropped_seg, cropped_depth
    
    def get_masked_depth_cropped(self, padding: int = 0) -> np.ndarray:
        """
        Get cropped depth with background masked out (set to NaN).
        
        Args:
            padding: Number of pixels to pad around the bounding box.
        
        Returns:
            Cropped depth array with background pixels set to NaN.
        """
        cropped_seg, cropped_depth = self.crop_to_object(padding)
        masked_depth = cropped_depth.copy()
        masked_depth[cropped_seg == 0] = np.nan
        return masked_depth
    
    def normalize_depth(self, method: str = "minmax") -> np.ndarray:
        """
        Normalize depth values within the object region.
        
        Args:
            method: Normalization method - "minmax" (0-1 range) or "zscore".
        
        Returns:
            Normalized depth array with background as NaN.
        """
        object_depth = self.get_object_depth()
        depth_values = self.get_object_depth_values()
        
        if len(depth_values) == 0:
            return object_depth
        
        if method == "minmax":
            min_val, max_val = np.min(depth_values), np.max(depth_values)
            if max_val - min_val > 0:
                normalized = (object_depth - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(object_depth)
                normalized[self.segmentation_mask == 0] = np.nan
        elif method == "zscore":
            mean_val, std_val = np.mean(depth_values), np.std(depth_values)
            if std_val > 0:
                normalized = (object_depth - mean_val) / std_val
            else:
                normalized = np.zeros_like(object_depth)
                normalized[self.segmentation_mask == 0] = np.nan
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def get_centroid(self) -> Optional[Tuple[float, float]]:
        """
        Calculate the centroid of the segmented object.
        
        Returns:
            Tuple (x, y) centroid coordinates or None if no object pixels.
        """
        indices = np.where(self.segmentation_mask == 1)
        if len(indices[0]) == 0:
            return None
        
        y_centroid = float(np.mean(indices[0]))
        x_centroid = float(np.mean(indices[1]))
        return x_centroid, y_centroid
    
    def get_depth_at_centroid(self) -> Optional[float]:
        """
        Get the depth value at the object's centroid.
        
        Returns:
            Depth value at centroid or None if no object pixels.
        """
        centroid = self.get_centroid()
        if centroid is None:
            return None
        
        x, y = int(round(centroid[0])), int(round(centroid[1]))
        return float(self.depth_mask[y, x])

