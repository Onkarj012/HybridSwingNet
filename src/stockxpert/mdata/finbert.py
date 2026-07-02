"""
FinBERT sentiment scoring using HuggingFace Transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from typing import List
import logging

logger = logging.getLogger("stockxpert.mdata.finbert")


class FinBERTScorer:
    """
    Sentiment scorer using ProsusAI/finbert model.
    Returns score = P(positive) - P(negative)
    """
    
    MODEL_NAME = "ProsusAI/finbert"
    
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        """
        Args:
            device: 'cuda', 'cpu', or 'mps'
            batch_size: Batch size for inference
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        logger.info(f"Loading FinBERT model on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("FinBERT model loaded")
    
    def score_headlines(self, headlines: List[str]) -> List[float]:
        """
        Score a list of headlines.
        
        Args:
            headlines: List of text strings
        
        Returns:
            List of sentiment scores (positive - negative probability)
        """
        if not headlines:
            return []
        
        scores = []
        
        # Process in batches
        for i in range(0, len(headlines), self.batch_size):
            batch = headlines[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
            # FinBERT outputs: [negative, neutral, positive]
            # Score = P(pos) - P(neg)
            batch_scores = (probs[:, 2] - probs[:, 0]).cpu().numpy().tolist()
            scores.extend(batch_scores)
        
        return scores
    
    def score_dataframe(self, df: pd.DataFrame, text_col: str = 'title') -> pd.DataFrame:
        """
        Add sentiment scores to DataFrame.
        
        Args:
            df: DataFrame with text column
            text_col: Name of text column
        
        Returns:
            DataFrame with added 'sentiment_score' column
        """
        if df.empty or text_col not in df.columns:
            df['sentiment_score'] = 0.0
            return df
        
        headlines = df[text_col].fillna('').tolist()
        scores = self.score_headlines(headlines)
        
        df = df.copy()
        df['sentiment_score'] = scores
        
        logger.info(f"Scored {len(scores)} headlines (mean={sum(scores)/len(scores):.3f})")
        return df
