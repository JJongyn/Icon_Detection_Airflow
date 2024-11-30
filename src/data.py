
import os
import random
import shutil
import logging
import numpy as np
import json
import yaml

from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple 

from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from .diffusion import AppIconGenerator
from .model import DetectionService
from .utils import ClassIDManager, generate_output_path, move_files, draw_labeled_bounding_boxes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        try:
            self.config = config
            self.class_id_manager = ClassIDManager()
            self.detector = DetectionService(self.config)
            logger.info("DataProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DataProcessor: {e}")
            raise
    
    def _calculate_bbox_center(self, bbox: List[Tuple[int, int]]) -> Tuple[float, float]:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        return (center_x, center_y)

    def _convert_to_yolo_format(self, bbox: List[float], image_width: int, image_height: int) -> List[float]:
        x_min, y_min, x_max, y_max = bbox
        center_x = ((x_min + x_max) / 2) / image_width
        center_y = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        
        return [center_x, center_y, width, height]
    
    def _match_bbox_by_proximity(self, ocr_results: List, detection_results: List, max_distance=100):
        # Extract OCR centers
        ocr_centers = [self._calculate_bbox_center(ocr.bbox) for ocr in ocr_results]
        yolo_data = []
        
        for result in detection_results[0]:
            bbox = result.boxes.xyxy[0].tolist()  
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            class_name = result.names[result.boxes.cls[0].item()]
            confidence = result.boxes.conf[0].item()
            
            yolo_data.append({
                'center': (center_x, center_y),
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
        
        yolo_centers = [data['center'] for data in yolo_data]
        
        if not ocr_centers or not yolo_centers:
            return []
        
        distances = cdist(ocr_centers, yolo_centers)
        
        matched_results = []
        for i, ocr_result in enumerate(ocr_results):
            min_distance_idx = np.argmin(distances[i])
            
            if distances[i][min_distance_idx] <= max_distance:
                matched_result = {
                    'text': ocr_result.text,
                    'ocr_bbox': ocr_result.bbox,
                    'ocr_confidence': 1.0,  # Placeholder, adjust as needed
                    'yolo_class': yolo_data[min_distance_idx]['class'],
                    'yolo_confidence': yolo_data[min_distance_idx]['confidence'],
                    'yolo_bbox': yolo_data[min_distance_idx]['bbox']
                }
                matched_results.append(matched_result)
        
        return matched_results
    
    def process_labeling(self, image_path: str, task_instance=None, output_path: str = None):
        try:
            logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>> Step 1: Preparing Data <<<<<<<<<<<<<<<<<<<<<<<")
            # 1: Get ocr and yolo results
            
            results = self.detector.process_image(image_path)
            ocr_results = results["ocr_results"]
            yolo_resutls = results["detection_results"] # # print(yolo_resutls)
            
            # 2: Match bounding boxes (center를 기준으로)
            matched_results = self._match_bbox_by_proximity(ocr_results, yolo_resutls)
            
            if self.config["draw_result"]:
                draw_labeled_bounding_boxes(image_path, 
                                            matched_results, 
                                            output_path=self.config["draw_path"],
                                            bbox_path=self.config["bbox_path"],)
            
            # 3: Convert Yolo Format
            yolo_data = []
            
            image_width = yolo_resutls[0].orig_shape[0]
            image_height = yolo_resutls[0].orig_shape[1]
        
            for result in matched_results:
                
                class_id = self.class_id_manager.get_class_id(result["text"])
                yolo_bbox = self._convert_to_yolo_format(result["yolo_bbox"], image_width, image_height)
                yolo_data.append([class_id] + yolo_bbox)
             
            if output_path is None:
                output_path = generate_output_path(image_path)
                
            # 4: Save to YOLO format file
            with open(output_path, "w") as file:
                for item in yolo_data:
                    file.write(" ".join(map(str, item)) + "\n")
                    

            logger.info(f"Processed image saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise
        

class DataAugmentor:
    def __init__(self, config):
        self.config = config
        self.split_raito = self.config["split_ratio"] # for training. ex) 80%
        self.aug_flag = self.config["aug_flag"]
        
        self.train_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.get("train_dir")))
        self.valid_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.get("valid_dir")))
        self.class_id_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.get("class_id")))
        self.data_yaml_path =  Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.get("yolo_data_path")))
        
        self.image_dirs = {
            "train": self.train_dir / "images",
            "valid": self.valid_dir / "images"
        }
        self.label_dirs = {
            "train": self.train_dir / "labels",
            "valid": self.valid_dir / "labels"
        }
        
        if self.aug_flag:
            controlnet_model = "lllyasviel/sd-controlnet-canny"
            diffusion_model = "CompVis/stable-diffusion-v1-4"
            device = "cpu"
            self.diffusion = AppIconGenerator(controlnet_model, diffusion_model, device)
            self.augmentation_path = self.config.get("augmentation_path")
            self.icon_path = self.config.get("bbox_path")
            os.makedirs(self.augmentation_path, exist_ok=True)
                    
    def _split_data(self, file_list, train_ratio=0.8):
        return train_test_split(file_list, train_size=train_ratio, random_state=42)

    def _move_files(self, file_list, target_dir, image_subdir="images", label_subdir="labels"):
        """
        파일을 지정된 폴더로 이동.
        :param file_list: 이미지 파일 리스트
        :param target_dir: 이동할 대상 디렉토리
        :param image_subdir: 이미지 하위 디렉토리 이름
        :param label_subdir: 라벨 하위 디렉토리 이름
        """
        for image_file in file_list:
            label_file = image_file.with_suffix('.txt')  # 이미지 파일에 대응하는 라벨 파일

            # 이미지 파일 이동
            target_image_dir = target_dir / image_subdir
            target_label_dir = target_dir / label_subdir

            target_image_path = target_image_dir / image_file.name
            
            shutil.move(str(image_file), str(target_image_path))
            logger.info(f"Moved image: {image_file.name} -> {target_image_dir}")

            # 라벨 파일 이동
            if label_file.exists():
                target_label_path = target_label_dir / label_file.name
                shutil.move(str(label_file), str(target_label_path))
                logger.info(f"Moved label: {label_file.name} -> {target_label_dir}")
            else:
                logger.warning(f"No label file found for image: {image_file.name}")
                
    def initialize_data(self):
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>> Step 2: Augmenting Data <<<<<<<<<<<<<<<<<<<<<<<")
        try:
        
            # logger.info(f"Reading class_id from {self.class_id_path}")
            with open(self.class_id_path, 'r', encoding='utf-8') as file:
                class_data = json.load(file)
                
            class_names = [None] * (max(class_data.values()) + 1)
            for name, id in class_data.items():
                class_names[id] = name

            # logger.info(f"Reading data.yaml from {self.data_yaml_path}")
            with open(self.data_yaml_path, 'r', encoding='utf-8') as file:
                data_yaml = yaml.safe_load(file)
            
            
            if len(class_names) == data_yaml.get('nc', 0):
                logger.info("No change in class count. Passing data files update.")
            else:
                logger.info(f"Updating data.yaml with {len(class_names)} classes.")
                data_yaml['nc'] = len(class_names)
                data_yaml['names'] = class_names

                # 4. data.yaml 파일 업데이트
                with open(self.data_yaml_path, 'w', encoding='utf-8') as file:
                    yaml.safe_dump(data_yaml, file, allow_unicode=True, default_flow_style=False)
                
                logger.info("Data.yaml has been updated with new class information.")
                
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
                        
    def run_augmentation(self, task_instance=None):
        """
        데이터 증강 실행
        :param task_instance: Airflow 작업 인스턴스
        """
        try:
            # data_path = task_instance.xcom_pull(task_ids='process_labeling_task', key='output_path')
            data_path = task_instance  # 테스트 데이터 경로
            
            # 증강 옵션이 꺼져 있으면 데이터 이동만 수행
            if not self.aug_flag:
                logger.info("Skipping Data Augmentation Step")
                # self._move_files(list(Path(data_path).glob("*.jpg")) + list(Path(data_path).glob("*.png")), self.train_dir)
                return
            
            # 데이터 증강 시작
            logger.info("Starting Data Augmentation Step")

            
            ### Diffusion based data augmentation ###
            self.diffusion.generate_icons_for_all_images(self.icon_path, self.augmentation_path)
            
            # 파일 가져오기
            image_files = list(Path(data_path).rglob("*.jpg"))
            if not image_files:
                logger.error("No image files found in the dataset.")
                return
            # 데이터 분할
            train_images, valid_images = self._split_data(image_files)

            # 학습 및 검증 데이터 이동
            self._move_files(train_images, self.train_dir)
            self._move_files(valid_images, self.valid_dir)

            logger.info("Data augmentation and split complete.")

        except Exception as e:
            logger.error(f"Error in run_augmentation: {str(e)}")