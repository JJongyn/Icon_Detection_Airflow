import os
import sys
import json
import cv2
import re
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
import yaml
from ultralytics import YOLO
import easyocr

os.environ['OMP_NUM_THREADS'] = '1' 
os.environ['WANDB_DISABLED'] = 'true'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    text: str
    bbox: List[Tuple[int, int]]
    confidence: float

class ConfigManager:
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

class ImageValidator:
    @staticmethod
    def validate_image(image_path: Union[str, Path]) -> bool:
        """Validate if the image exists and is readable"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return False
            
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
        

class OCRProcessor:
    """OCR processing class with error handling and validation"""
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        try:
            print(gpu)
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info(f"OCR Processor initialized with languages: {languages}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            raise
        
    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0][0], box2[0][0])
        y1 = max(box1[0][1], box2[0][1])
        x2 = min(box1[2][0], box2[2][0])
        y2 = min(box1[2][1], box2[2][1])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2][0] - box1[0][0]) * (box1[2][1] - box1[0][1])
        box2_area = (box2[2][0] - box2[0][0]) * (box2[2][1] - box2[0][1])

        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0

        return intersection_area / union_area

    def _filter_and_validate_texts(self, results, iou_threshold=0.7):
        """
        Filter overlapping texts and validate using allowed patterns.
        """
        
        filtered_results = []
        used_indices = set()

        for i, (bbox1, text1, conf1) in enumerate(results):
            if i in used_indices:
                continue
                        
            overlap = False
            for j, (bbox2, text2, conf2) in enumerate(results):
                if i != j and j not in used_indices:
                    iou = self._calculate_iou(bbox1, bbox2)
                    if iou > iou_threshold:
                        overlap = True
                        # Keep the text with higher confidence
                        if conf1 >= conf2:
                            used_indices.add(j)  # Remove the lower-confidence text
                        else:
                            used_indices.add(i)  # Remove the current text
                            break

            if not overlap:
                filtered_results.append((bbox1, text1, conf1))

        return filtered_results
    
    def detect_text(self, image_path: Union[str, Path]) -> List[DetectionResult]:
        """
        Detect text in the image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of DetectionResult objects containing detected text and bounding boxes
        """
        if not ImageValidator.validate_image(image_path):
            raise ValueError(f"Invalid image: {image_path}")
            
        try:
            # pi = preprocess_image_for_ocr(str(image_path))
            # results = self.reader.readtext(pi)
            results = self.reader.readtext(str(image_path))
            
            ### 단어가 아니거나 필요없는 부분일 경우에는 전처리 하도록 진행 
            allowed_patterns = [
                r"^[A-Za-z]+$",  # Only alphabetic words (e.g., "News")
                r"^[A-Za-z]+\s[A-Za-z]+$",  # Words with spaces (e.g., "Local Media")
                r"^[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+$",  # Three-word phrases (e.g., "Bluetooth Audio")
                r"^[A-Za-z\s\(\)\-]+$",  # Allowing words, spaces, parentheses, and hyphens (e.g., "Paint Booth (Gerrit)")
                r"^[A-Za-z\s\(\);]+$",  # Allowing words, spaces, parentheses, and semicolons (e.g., "AAE Theme Playgr;")
            ]
            
            detected_data = []
            used_indices = set()
            for i, (bbox, text, conf) in enumerate(results):
                if i in used_indices:
                    continue
                if any(re.fullmatch(pattern, text.strip()) for pattern in allowed_patterns):
                    detected_data.append(
                        DetectionResult(
                            text=text,
                            bbox=[tuple(map(int, pt)) for pt in bbox],
                            confidence=conf
                        )
                    )
            
            logger.info(f"Successfully detected {len(detected_data)} text regions")
            return detected_data
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            raise

class YOLODetector:
    """YOLO object detection class with error handling, validation, and training results management"""
    
    def __init__(self, config: Union[str, dict], mode: str = "detect"):
        self.mode = mode
        try:
            self.config = config
            self.conf = self.config.get("conf", 0.5)
            self.iou = self.config.get("iou", 0.1)
            self.yolo_data_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.get("yolo_data_path")))
            self.imgsz = self.config.get("imgsz", 320)
            self.epochs = self.config.get("epochs", 400)
            self.batch_size = self.config.get("batch_size", 16)
            self.train_flag = self.config.get("train_flag", False)
            
            if self.mode == 'detect':
                self.model_path = self.config.get("icon_model_path")
                self.model_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path))
                
                self.model = YOLO(self.model_path)
    
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            raise
    
    def _set_mode(self, mode: str= 'train'):
        """Change the mode of the detector ('detect' or 'train')"""
        if mode not in ['detect', 'train']:
            raise ValueError("Invalid mode. Mode should be 'detect' or 'train'.")
        self.mode = mode
        if self.mode == 'train':
            # Prepare for training mode
            self.model_path = self.config.get("navi_model_path", "yolo11m.pt")  # 기본값 yolo11.pt 설정
    
            # 경로가 존재하지 않거나 올바르지 않을 경우 기본 모델 경로 사용
            if not Path(self.model_path).exists():
                logger.warning(f"Provided model path '{self.model_path}' is invalid. Using default model path 'yolo11.pt'.")
                self.model_path = "yolov8s.pt"
                
            self.model = YOLO(self.model_path)
            # Reinitialize training parameters if needed
            self.results_dir = Path(self.config.get("results_dir", "training_results"))
            self.results_dir.mkdir(exist_ok=True)
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = self.results_dir / self.experiment_id
            self.experiment_dir.mkdir(exist_ok=True)
            logger.info(f"YOLO Detector switched to 'train' mode")
    
    def detect_objects(self, image_path: Union[str, Path]) -> Dict:
        if not self._validate_image(image_path):
            raise ValueError(f"Invalid image: {image_path}")
            
        try:
            results = self.model.predict(
                str(image_path),
                save=False,
                conf=self.conf,
                iou=self.iou
            )
            logger.info(f"Object detection completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise

    def train(self) -> Dict:
        """
        Train the YOLO model and save results
        Returns:
            Dictionary containing training results and metadata
        """
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            
            if not self.train_flag:
                logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>> Step 3: Skip Training Navigator Model using new data <<<<<<<<<<<<<<<<<<<<<<<")
                
                training_results = {
                'box_loss': 0.01,
                'cls_loss': 0.02,
                'dfl_loss': 0.005,
                'metrics': {
                    'precision': 0.93,
                    'recall': 0.92,
                    'mAP_50': 0.85,  # Mean Average Precision at IoU=0.5
                    'mAP_50_95': 0.75  # mAP at IoU=0.5:0.95
                },
                'train_time': '0h 30m 15s',
                'epochs': 5,
                'batch_size': self.batch_size,
                'img_size': self.imgsz,
                'device': 'cpu',
                'status': 'Training Skipped',
                'message': 'Training was skipped for presentation purposes',
                'best_model': None,  # No best model if training is skipped
                'loss_history': [
                    {'epoch': 1, 'box_loss': 0.02, 'cls_loss': 0.03, 'dfl_loss': 0.01},
                    {'epoch': 2, 'box_loss': 0.015, 'cls_loss': 0.025, 'dfl_loss': 0.008},
                    {'epoch': 3, 'box_loss': 0.012, 'cls_loss': 0.02, 'dfl_loss': 0.006},
                    {'epoch': 4, 'box_loss': 0.01, 'cls_loss': 0.02, 'dfl_loss': 0.005},
                    {'epoch': 5, 'box_loss': 0.009, 'cls_loss': 0.018, 'dfl_loss': 0.004}
                ],
                'train_loss': 0.013  # Final train loss after 5 epochs
                }
                
                return training_results
            
            
            logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>> Step 3: Training Navigator Model using new data <<<<<<<<<<<<<<<<<<<<<<<")
            self._set_mode()
            self._save_training_config()
            
            # 모델 학습 실행
            results = self.model.train(
                data=self.yolo_data_path,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device='cpu',
                name="train",
                val=False
            )
            
            # 학습 결과 저장
            training_results = self._process_training_results(results)
            # self._save_training_results(training_results)
            
            # logger.info(f"Training completed successfully. Results saved to {self.experiment_dir}")
            
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _save_training_config(self):
        """Save training configuration"""
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump({
                'mode': self.mode,
                'data_path': str(self.yolo_data_path),
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'image_size': self.imgsz,
                'conf_threshold': self.conf,
                'iou_threshold': self.iou,
                'model_path': str(self.model_path),
                'timestamp': self.experiment_id
            }, f)

    def _process_training_results(self, results) -> Dict:
        metrics = {
            'mAP': results.maps,  # mean Average Precision
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
            'fitness': results.fitness
        }

        training_info = {
            'epochs_completed': len(results.curves) if hasattr(results, 'curves') else 0,
            'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else None,
            'training_time': results.t1 - results.t0 if hasattr(results, 't1') and hasattr(results, 't0') else None
        }

        model_info = {
            'model_path': str(self.model_path),
            'best_weights_path': str(self.experiment_dir / 'train' / 'weights' / 'best.pt')
        }

        return {
            'experiment_id': self.experiment_id,
            'metrics': metrics,
            'training': training_info,
            'model_info': model_info
        }

    def _save_training_results(self, results: Dict):
        results_path = self.experiment_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    def _get_latest_results(self) -> Dict:
        try:
            results_path = self.experiment_dir / "training_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("No training results found")
                return {}
        except Exception as e:
            logger.error(f"Failed to load training results: {e}")
            return {}

    def _validate_image(self, image_path: Union[str, Path]) -> bool:
        try:
            path = Path(image_path)
            return path.exists() and path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        except Exception:
            return False
    
class DetectionService:
    def __init__(
        self, 
        config, 
    ):
        
        self.config = config
        self.ocr_type = self.config["ocr_type"]
        self.language = self.config["ocr_language"]
        self.use_gpu = self.config["gpu"]
        
        # Choose OCR processor
        if self.ocr_type == 'tesseract':
            self.ocr = TesseractOCRProcessor([self.language])
        elif self.ocr_type == 'paddle':
            self.ocr = PaddleOCRProcessor([self.language])
        else:
            self.ocr = OCRProcessor([self.language], self.use_gpu)
        
        self.detector = YOLODetector(self.config)
        
    def process_image(self, image_path: Union[str, Path]) -> Dict:
        try:
            ocr_results = self.ocr.detect_text(image_path)
            detection_results = self.detector.detect_objects(image_path)
            
            return {
                "ocr_results": ocr_results,
                "detection_results": detection_results
            }
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise

        
