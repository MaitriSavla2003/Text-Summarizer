from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_evaluation import ModelEvaluation
from src.textSummarizer.logging import logger
from datasets import load_metric



class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
    
        # Load the metric with trust_remote_code=True
        rouge_metric = load_metric('rouge', trust_remote_code=True)
    
        model_evaluation.evaluate()