"""
Utility functions for geometry_tools visualization and testing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def draw_lines_on_image(image: np.ndarray, lines: List[Dict], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw detected lines on an image.
    
    Args:
        image: Input image
        lines: List of line dictionaries with coordinates
        color: RGB color tuple (default: green)
        thickness: Line thickness
    
    Returns:
        Image with lines drawn
    """
    result = image.copy()
    for line in lines:
        if 'coordinates' in line:
            x1, y1, x2, y2 = line['coordinates']
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return result


def highlight_line(image: np.ndarray, line_coords: Tuple[float, float, float, float], 
                  color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> np.ndarray:
    """
    Highlight a specific line on an image.
    
    Args:
        image: Input image
        line_coords: (x1, y1, x2, y2) coordinates
        color: RGB color tuple (default: red)
        thickness: Line thickness
    
    Returns:
        Image with highlighted line
    """
    result = image.copy()
    x1, y1, x2, y2 = line_coords
    cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return result


def save_visualization(image: np.ndarray, filename: str, dpi: int = 150) -> None:
    """
    Save image visualization to file.
    
    Args:
        image: Image to save
        filename: Output filename
        dpi: Resolution for saving
    """
    plt.figure(figsize=(10, 8))
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_comparison_visualization(original_image: np.ndarray, processed_image: np.ndarray, 
                                   title: str = "Line Detection Comparison") -> None:
    """
    Create side-by-side comparison of original and processed images.
    
    Args:
        original_image: Original input image
        processed_image: Processed image with detections
        title: Title for the comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    if len(original_image.shape) == 3:
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Processed image
    if len(processed_image.shape) == 3:
        ax2.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(processed_image, cmap='gray')
    ax2.set_title('Detected Lines')
    ax2.axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def create_synthetic_lines(width: int = 800, height: int = 400, 
                          line_configs: Optional[List[Dict]] = None) -> np.ndarray:
    """
    Create synthetic image with lines for testing.
    
    Args:
        width: Image width
        height: Image height
        line_configs: List of line configurations with 'start', 'end', 'color', 'thickness'
    
    Returns:
        Synthetic image with lines
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    if line_configs is None:
        # Default: two horizontal lines of different lengths
        line_configs = [
            {'start': (100, 150), 'end': (300, 150), 'color': (0, 0, 0), 'thickness': 3},
            {'start': (450, 250), 'end': (750, 250), 'color': (0, 0, 0), 'thickness': 3}
        ]
    
    for config in line_configs:
        cv2.line(image, config['start'], config['end'], config['color'], config['thickness'])
    
    return image


def create_muller_lyer_illusion(shaft_length: int = 200, arrow_size: int = 30, 
                               image_size: Tuple[int, int] = (800, 400)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Müller-Lyer illusion images (inward and outward arrows).
    
    Args:
        shaft_length: Length of the main line shaft
        arrow_size: Size of arrow heads
        image_size: (width, height) tuple
    
    Returns:
        Tuple of (inward_arrows_image, outward_arrows_image)
    """
    width, height = image_size
    
    # Create blank images
    img_inward = np.ones((height, width, 3), dtype=np.uint8) * 255
    img_outward = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Calculate positions
    center_y = height // 2
    left_x = width // 4
    right_x = 3 * width // 4
    
    # Draw main shafts
    cv2.line(img_inward, (left_x - shaft_length//2, center_y), 
            (left_x + shaft_length//2, center_y), (0, 0, 0), 3)
    cv2.line(img_outward, (right_x - shaft_length//2, center_y), 
            (right_x + shaft_length//2, center_y), (0, 0, 0), 3)
    
    # Draw inward arrows (pointing toward center)
    # Left line arrows
    cv2.line(img_inward, (left_x - shaft_length//2, center_y), 
            (left_x - shaft_length//2 + arrow_size, center_y - arrow_size), (0, 0, 0), 3)
    cv2.line(img_inward, (left_x - shaft_length//2, center_y), 
            (left_x - shaft_length//2 + arrow_size, center_y + arrow_size), (0, 0, 0), 3)
    cv2.line(img_inward, (left_x + shaft_length//2, center_y), 
            (left_x + shaft_length//2 - arrow_size, center_y - arrow_size), (0, 0, 0), 3)
    cv2.line(img_inward, (left_x + shaft_length//2, center_y), 
            (left_x + shaft_length//2 - arrow_size, center_y + arrow_size), (0, 0, 0), 3)
    
    # Draw outward arrows (pointing away from center)
    # Right line arrows
    cv2.line(img_outward, (right_x - shaft_length//2, center_y), 
            (right_x - shaft_length//2 - arrow_size, center_y - arrow_size), (0, 0, 0), 3)
    cv2.line(img_outward, (right_x - shaft_length//2, center_y), 
            (right_x - shaft_length//2 - arrow_size, center_y + arrow_size), (0, 0, 0), 3)
    cv2.line(img_outward, (right_x + shaft_length//2, center_y), 
            (right_x + shaft_length//2 + arrow_size, center_y - arrow_size), (0, 0, 0), 3)
    cv2.line(img_outward, (right_x + shaft_length//2, center_y), 
            (right_x + shaft_length//2 + arrow_size, center_y + arrow_size), (0, 0, 0), 3)
    
    return img_inward, img_outward


def print_detection_summary(detection_result: Dict) -> None:
    """
    Print a formatted summary of line detection results.
    
    Args:
        detection_result: Result from line detection functions
    """
    if not detection_result.get('success', False):
        print(f"❌ Detection failed: {detection_result.get('error', 'Unknown error')}")
        return
    
    lines = detection_result.get('lines', [])
    print(f"✅ Detection successful: Found {len(lines)} lines")
    print("\n📏 Line Details:")
    for i, line in enumerate(lines):
        length = line.get('length', 0)
        angle = line.get('angle', 0)
        coords = line.get('coordinates', (0, 0, 0, 0))
        print(f"  Line {i}: {length:.1f}px, angle: {angle:.1f}°, coords: {coords}")


def print_comparison_result(comparison_result: Dict) -> None:
    """
    Print a formatted summary of line comparison results.
    
    Args:
        comparison_result: Result from comparison functions
    """
    if not comparison_result.get('success', False):
        print(f"❌ Comparison failed: {comparison_result.get('error', 'Unknown error')}")
        return
    
    answer = comparison_result.get('answer_text', 'unknown')
    confidence = comparison_result.get('confidence', 'unknown')
    line_a_length = comparison_result.get('line_A_length', 0)
    line_b_length = comparison_result.get('line_B_length', 0)
    difference = comparison_result.get('difference', 0)
    
    print(f"✅ Comparison successful")
    print(f"📊 Answer: {answer} (confidence: {confidence})")
    print(f"📏 Line A: {line_a_length:.1f}px")
    print(f"📏 Line B: {line_b_length:.1f}px")
    print(f"📐 Difference: {difference:.1f}px")
    
    if comparison_result.get('are_equal', False):
        print("✅ Lines are approximately equal (within tolerance)")
