import json
import os
import cv2
import numpy as np
import yaml
import logging
import shutil

from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple 

from airflow.hooks.base_hook import BaseHook

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management class"""
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

class ClassIDManager:
    def __init__(self, file_path: Union[str, Path] = 'class_id.json'):
        """
        Initialize ClassIDManager with the path to the JSON file.
        
        Args:
            file_path: Path to the JSON file storing class-ID mappings.
        """
        # self.file_path = Path(file_path)
        self.file_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', file_path))
        
        if not self.file_path.exists():
            # Initialize an empty dictionary if the file doesn't exist
            with open(self.file_path, "w") as file:
                json.dump({}, file)
        self.load_mappings()

    def load_mappings(self):
        """Load class-ID mappings from the JSON file."""
        with open(self.file_path, "r") as file:
            self.class_id_mapping = json.load(file)

    def save_mappings(self):
        """Save class-ID mappings to the JSON file."""
        with open(self.file_path, "w") as file:
            json.dump(self.class_id_mapping, file, indent=4)

    def get_class_id(self, class_name: str) -> int:
        """
        Get the ID for a given class name, adding it if it doesn't exist.
        
        Args:
            class_name: The name of the class.
        
        Returns:
            The ID corresponding to the class name.
        """
        if class_name not in self.class_id_mapping:
            # Assign a new ID to the class name
            new_id = len(self.class_id_mapping)
            self.class_id_mapping[class_name] = new_id
            self.save_mappings()
        return self.class_id_mapping[class_name]

    def get_all_mappings(self) -> Dict[str, int]:
        """
        Get all class-ID mappings.
        
        Returns:
            Dictionary of all class-ID mappings.
        """
        return self.class_id_mapping

def move_files(image_files, target_dir, image_subdir, label_subdir):
        """
        Moves the image and label files into the specified target directory.
        Assumes the image and label files share the same names (but different extensions).
        """
        for image_file in image_files:
            label_file = Path(str(image_file).replace('.jpg', '.txt'))  # Assuming labels are .txt

            # Move image to target directory
            image_target = target_dir / image_subdir / image_file.name
            shutil.move(image_file, image_target)

            # Move label to target directory
            label_target = target_dir / label_subdir / label_file.name
            if label_file.exists():
                shutil.move(label_file, label_target)
            else:
                logger.warning(f"Label file not found for image: {image_file.name}")

def generate_output_path(image_path: str) -> str:
    """
    Generate an output path based on the image file name, in the same directory.
    """
    image_dir = os.path.dirname(image_path)  # Get the directory of the image
    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)  # Remove file extension
    output_path = os.path.join(image_dir, f"{name}.txt")  # Generate the output path
    return output_path


def draw_labeled_bounding_boxes(
    image_path: str, 
    matched_results: List[Dict], 
    output_path: str = None
) -> np.ndarray:
    """
    Draw labeled bounding boxes on the original image
    
    Args:
        image_path: Path to the original image
        matched_results: List of matched detection results
        output_path: Optional path to save the annotated image
    
    Returns:
        Annotated image as numpy array
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Define color palette
        colors = {
            'default': (0, 255, 0),  # Green for default
            'text': (255, 0, 0),     # Blue for text
            'object': (0, 0, 255)    # Red for object
        }
        
        # Draw bounding boxes with labels
        for result in matched_results:
            # YOLO bbox coordinates
            bbox = [int(x) for x in result['yolo_bbox']]
            x1, y1, x2, y2 = bbox
            
            # Choose color based on confidence
            color = colors['default']
            if result.get('yolo_confidence', 0) > 0.7:
                color = colors['default']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            # label = f"{result['yolo_class']}: {result['text']}"
            label = f"{result['text']}"
            conf = result.get('yolo_confidence', 0)
            # full_label = f"{label} ({conf:.2f})"
            full_label = f"{label}"
            
            # Add label text
            cv2.putText(
                image, 
                full_label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )
        
        # Save image if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    
    except Exception as e:
        logging.error(f"Error drawing bounding boxes: {e}")
        raise

def start():
    return logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>> Step 0: Initialization Processor <<<<<<<<<<<<<<<<<<<<<<<")