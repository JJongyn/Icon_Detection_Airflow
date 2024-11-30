import os
import logging

from typing import List, Dict, Tuple

from .model import DetectionService, YOLODetector
from .data import DataProcessor, DataAugmentor
from .llm import AIAgent
from .utils import ConfigManager, start

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MobisMain(DataProcessor, DataAugmentor, YOLODetector, AIAgent):
    def __init__(self, config="en_yolo11.yaml"):
        start()
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', config)
        self.config = ConfigManager(config_path).config
        DataProcessor.__init__(self, self.config)
        DataAugmentor.__init__(self, self.config)
        YOLODetector.__init__(self, self.config)
        AIAgent.__init__(self, self.config)
    
    # 새로운 버전의 데이터 검출 및 자동 라벨링    
    def prepare_data(self, image_path: str, **kwargs):
        output_path = self.process_labeling(image_path)
        return output_path
        
    # 데이터 증강 (기존 방법 + Diffusion based)
    def augment_data(self, task_instance, **kwargs):
        self.initialize_data() 
        self.run_augmentation(task_instance=task_instance)
    
    # Detector 학습
    def train_model(self, **kwargs):
        training_results = self.train() 
        return training_results
    
    # report 생성 (llm based)
    def report_result(self, results, **kwargs):
        response = self.run_report(results)
        return response
        
        