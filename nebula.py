from transformers import pipeline
import torch

class NebulaFunc:
    def __init__(self):
        print("Initializing Nebula with Hugging Face pipeline...")
        # Use Hugging Face's pipeline for sentiment analysis
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        print("Sentiment analysis pipeline loaded successfully")

    def sentimentAn(self, text):
        try:
            print(f"Analyzing sentiment for text: {text}")
            result = self.sentiment_pipeline(text)[0]
            print(f"Analysis result: {result}")
            
            if result['label'] == 'POSITIVE':
                return "The user seems to be happy"
            else:
                return "The user seems to be sad"


                
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return f"Error in sentiment analysis: {str(e)}"

    def sentimentMod(self):
        try:
            if self.sentiment_pipeline is not None:
                return "Sentiment model is loaded"
            else:
                return "Error in loading the model"
        except Exception as e:
            return f"Error in model loading: {str(e)}"
