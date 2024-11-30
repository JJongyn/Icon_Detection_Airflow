import openai

class AIAgent:
    def __init__(self, config):
        self.config = config
        self.prompt = self.config.get("prompt", "")    
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "gpt-3.5-turbo")
        self.temperature = self.config.get("temperature", 0.7)
        
        if self.api_key:
            openai.api_key = self.api_key
        else:
            raise ValueError("API key is required")
    
    def _analyze_yolo_results(self, results):
        analyzed = {
            "metrics": {},
            "improvements": [],
            "issues": [],
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "train_loss": 0.0
        }
        if "metrics" in results:
            metrics = results["metrics"]
            analyzed["metrics"] = {
                "mAP_50_95": metrics.get("mAP_50_95", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0)
            }
        
        if 'train_loss' in results:
            analyzed["train_loss"] = results.get("train_loss", None)
            analyzed["box_loss"] = results.get("box_loss", None)
            analyzed["cls_loss"] = results.get("cls_loss", None)
                
        return analyzed
    
    def _generate_text(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"
        
    def initialize_prompt(self):
        self.prompt = self.config.get("prompt", "")    
    
    def update_prompt(self, prompt=""):
        self.prompt = prompt
        
    def run_report(self, results):
        analyzed = self._analyze_yolo_results(results)
        
        report_prompt = f"{self.prompt}\n\n"
    
        metrics = analyzed["metrics"]
        report_prompt += f"- mAP: {metrics.get('mAP_50_95', 0):.3f}\n"
        report_prompt += f"- Precision: {metrics.get('precision', 0):.3f}\n"
        report_prompt += f"- Recall: {metrics.get('recall', 0):.3f}\n\n"
        report_prompt += f"- Box loss: {metrics.get('box_loss', 0):.3f}\n\n"
        report_prompt += f"- Class loss: {metrics.get('cls_loss', 0):.3f}\n\n"
        report_prompt += f"- Train loss: {metrics.get('train_loss', 0):.3f}\n\n"
        
        response = self._generate_text(report_prompt)
        
        return response
        
    
    