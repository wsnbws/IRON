#!/usr/bin/env python3
"""
Custom visualization class for segmentation model results.
Supports visualization of segmentation masks, predicted points, and confidence scores.
"""

import cv2
import numpy as np
import os
from typing import Optional, List, Tuple, Union


class SegmentationVisualizer:
    """
    Visualizer for segmentation model outputs with configurable display options.
    
    Features:
    - Segmentation mask overlay
    - Predicted point visualization with confidence-based brightness
    - Flexible control over visualization components
    """
    
    def __init__(
        self,
        show_mask: bool = True,
        show_points: bool = True,
        show_confidence: bool = True,
        show_info_text: bool = True,
        palette: Optional[List[List[int]]] = None,
        point_color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
        point_radius: int = 5,
        mask_alpha: float = 0.5,
        confidence_range: Tuple[float, float] = (0.3, 1.0)
    ):
        """
        Initialize the visualizer with display options.
        
        Args:
            show_mask (bool): Whether to show segmentation mask overlay
            show_points (bool): Whether to show predicted points
            show_confidence (bool): Whether to modulate point brightness by confidence
            show_info_text (bool): Whether to show coordinate and confidence text overlay
            palette (List[List[int]]): Color palette for segmentation classes (BGR format)
            point_color (Tuple[int, int, int]): Base color for points in BGR format
            point_radius (int): Radius of point markers
            mask_alpha (float): Alpha blending factor for mask overlay (0.0-1.0)
            confidence_range (Tuple[float, float]): Min/max brightness range for confidence
        """
        self.show_mask = show_mask
        self.show_points = show_points
        self.show_confidence = show_confidence
        self.show_info_text = show_info_text
        self.palette = palette or [[128, 0, 0], [0, 128, 0]]  # Default: background, foreground
        self.point_color = point_color
        self.point_radius = point_radius
        self.mask_alpha = mask_alpha
        self.confidence_range = confidence_range
        
        # Convert palette to numpy array for efficient indexing
        self.palette_array = np.array(self.palette, dtype=np.uint8)
        
        # Text display settings
        self.show_info_text = show_info_text
        self.text_color = (0, 0, 0)        # Black text
        self.text_bg_color = (255, 255, 255)  # White background
        self.text_bg_alpha = 0.7           # Semi-transparent background
        self.text_font_scale = 0.6
        self.text_thickness = 1
        self.text_margin = 10
    
    def _create_mask_overlay(
        self, 
        img: np.ndarray, 
        seg_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create colored segmentation mask overlay.
        
        Args:
            img (np.ndarray): Input image in BGR format (H, W, 3)
            seg_mask (np.ndarray): Segmentation mask with class labels (H, W)
            
        Returns:
            np.ndarray: Image with mask overlay applied
        """
        H, W = seg_mask.shape
        color_mask = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Apply colors for each class
        for class_id in range(len(self.palette)):
            mask_pixels = (seg_mask == class_id)
            if mask_pixels.any():
                color_mask[mask_pixels] = self.palette_array[class_id]
        
        # Blend with original image
        blended = cv2.addWeighted(
            img, 1.0 - self.mask_alpha,
            color_mask, self.mask_alpha,
            0
        )
        
        return blended
    
    def _draw_points_with_confidence(
        self,
        img: np.ndarray,
        points: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw predicted points with confidence-modulated brightness.
        
        Args:
            img (np.ndarray): Input image in BGR format (H, W, 3)
            points (np.ndarray): Point coordinates (N, 2) in (x, y) format
            confidence (np.ndarray, optional): Confidence scores (N,) in range [0, 1]
            
        Returns:
            np.ndarray: Image with points drawn
        """
        img_with_points = img.copy()
        
        if len(points) == 0:
            return img_with_points
        
        # Handle confidence modulation
        if self.show_confidence and confidence is not None:
            # Normalize confidence to brightness range
            conf_min, conf_max = self.confidence_range
            normalized_conf = np.clip(confidence, 0.0, 1.0)
            brightness_factors = conf_min + (conf_max - conf_min) * normalized_conf
        else:
            # Use maximum brightness if confidence not shown
            brightness_factors = np.ones(len(points))
        
        # Draw each point with appropriate brightness
        for i, (point, brightness) in enumerate(zip(points, brightness_factors)):
            x, y = int(point[0]), int(point[1])
            
            # Skip points outside image bounds
            if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
                continue
            
            # Modulate color brightness based on confidence
            modulated_color = tuple(int(c * brightness) for c in self.point_color)
            
            # Draw filled circle for point
            cv2.circle(
                img_with_points,
                (x, y),
                self.point_radius,
                modulated_color,
                -1  # Filled circle
            )
            
            # Add white border for better visibility
            cv2.circle(
                img_with_points,
                (x, y),
                self.point_radius + 1,
                (255, 255, 255),
                1  # Border thickness
            )
        
        return img_with_points
    
    def _draw_info_text(
        self,
        img: np.ndarray,
        points: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw information text (coordinates and confidence) in the top-left corner.
        
        Args:
            img (np.ndarray): Input image in BGR format (H, W, 3)
            points (np.ndarray, optional): Point coordinates (N, 2) in (x, y) format
            confidence (np.ndarray, optional): Confidence scores (N,)
            
        Returns:
            np.ndarray: Image with information text drawn
        """
        if not self.show_info_text or (points is None and confidence is None):
            return img
            
        img_with_text = img.copy()
        
        # Prepare text lines
        text_lines = []
        
        if points is not None and len(points) > 0:
            text_lines.append("Point Coordinates:")
            for i, point in enumerate(points):
                x, y = point[0], point[1]
                coord_text = f"  Point {i+1}: ({x:.1f}, {y:.1f})"
                text_lines.append(coord_text)
        
        if confidence is not None and len(confidence) > 0:
            if text_lines:  # Add separator if we already have coordinate info
                text_lines.append("")
            text_lines.append("Confidence Scores:")
            for i, conf in enumerate(confidence):
                conf_text = f"  Point {i+1}: {conf:.3f}"
                text_lines.append(conf_text)
        
        if not text_lines:
            return img_with_text
        
        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = int(25 * self.text_font_scale)
        max_width = 0
        
        for line in text_lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line, font, self.text_font_scale, self.text_thickness
            )
            max_width = max(max_width, text_width)
        
        # Calculate background rectangle
        bg_width = max_width + 2 * self.text_margin
        bg_height = len(text_lines) * line_height + 2 * self.text_margin
        
        # Create semi-transparent background
        overlay = img_with_text.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (bg_width, bg_height),
            self.text_bg_color,
            -1
        )
        
        # Blend with original image
        cv2.addWeighted(
            overlay, self.text_bg_alpha,
            img_with_text, 1 - self.text_bg_alpha,
            0, img_with_text
        )
        
        # Draw text lines
        for i, line in enumerate(text_lines):
            y_pos = self.text_margin + (i + 1) * line_height - 5
            cv2.putText(
                img_with_text,
                line,
                (self.text_margin, y_pos),
                font,
                self.text_font_scale,
                self.text_color,
                self.text_thickness,
                cv2.LINE_AA
            )
        
        return img_with_text
    
    def display(
        self,
        img: Union[str, np.ndarray],
        seg_mask: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        out_file: Optional[str] = None,
    ) -> np.ndarray:
        """
        Main visualization function combining all display components.
        
        Args:
            img (str or np.ndarray): Input image path or BGR image array
            seg_mask (np.ndarray, optional): Segmentation mask (H, W) with class labels
            points (np.ndarray, optional): Predicted points (N, 2) in (x, y) format
            confidence (np.ndarray, optional): Point confidence scores (N,)
            show (bool): Whether to display result in window
            out_file (str, optional): Output file path to save result (directory will be created if not exists)
            win_name (str): Window name for display
            
        Returns:
            np.ndarray: Visualization result image
        """
        # Load image if path provided
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            assert img is not None, f"Could not load image from {img}"
        else:
            img = img.copy()
        
        result_img = img.copy()
        
        # Apply segmentation mask overlay
        if self.show_mask and seg_mask is not None:
            result_img = self._create_mask_overlay(result_img, seg_mask)
        
        # Draw predicted points with confidence
        if self.show_points and points is not None:
            result_img = self._draw_points_with_confidence(
                result_img, points, confidence
            )
        
        # Draw information text overlay
        result_img = self._draw_info_text(result_img, points, confidence)
        
        # Save result
        if out_file is not None:
            # Create directory if it doesn't exist
            out_dir = os.path.dirname(out_file)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            
            success = cv2.imwrite(out_file, result_img)
            if not success:
                raise ValueError(f"Failed to save image to {out_file}")
        
        return result_img
    
    def update_config(
        self,
        show_mask: Optional[bool] = None,
        show_points: Optional[bool] = None,
        show_confidence: Optional[bool] = None,
        show_info_text: Optional[bool] = None,
        **kwargs
    ):
        """
        Update visualization configuration.
        
        Args:
            show_mask (bool, optional): Update mask display setting
            show_points (bool, optional): Update points display setting
            show_confidence (bool, optional): Update confidence display setting
            show_info_text (bool, optional): Update info text display setting
            **kwargs: Additional configuration parameters
        """
        if show_mask is not None:
            self.show_mask = show_mask
        if show_points is not None:
            self.show_points = show_points
        if show_confidence is not None:
            self.show_confidence = show_confidence
        if show_info_text is not None:
            self.show_info_text = show_info_text
        
        # Update other parameters if provided
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Example usage and testing
if __name__ == "__main__":
    # Create visualizer with custom settings
    visualizer = SegmentationVisualizer(
        show_mask=True,
        show_points=True,
        show_confidence=True,
        palette=[[128, 0, 0], [0, 128, 0]],  # Background: dark red, Foreground: dark green
        point_color=(0, 0, 255),  # Red points
        point_radius=6,
        mask_alpha=0.4
    )
    
    print("SegmentationVisualizer initialized successfully!")
    print(f"Configuration: mask={visualizer.show_mask}, "
          f"points={visualizer.show_points}, "
          f"confidence={visualizer.show_confidence}")
