import os
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import string
import re
import numpy
import dotenv
import requests
import time

dotenv.load_dotenv()


class FraudDetectionHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FraudDetectionHandler, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.embedding_model = 'all-mpnet-base-v2'
        try:
            self.fraud_email_model = xgb.XGBClassifier()
            self.fraud_email_model.load_model(
                'models/xgb_fraud_email_model_768.json')

            self.fraud_sms_model = xgb.XGBClassifier()
            self.fraud_sms_model.load_model(
                'models/xgb_fraud_sms_model_768.json')

            self.fraud_url_model = xgb.XGBClassifier()
            self.fraud_url_model.load_model(
                'models/xgb_phishing_url_model_768.json')

            self.embedder = SentenceTransformer(self.embedding_model)
            # print("✅ All models loaded successfully into memory")
        except Exception as e:
            print(f"⚠️ An error occurred while loading models: {e}")
            self.fraud_email_model = None
            self.fraud_sms_model = None
            self.fraud_url_model = None
            self.embedder = None

    def check_fraud_email(self, email_text: str) -> bool:
        try:
            embedding = self.generate_embedding(self.clean_text(email_text))
            prediction = self.fraud_email_model.predict(
                embedding.reshape(1, -1))
            return prediction[0] == 1
        except Exception as e:
            print(f"⚠️ Error during fraud email check: {str(e)}")
            return False

    def check_fraud_sms(self, sms_text: str) -> bool:
        try:
            embedding = self.generate_embedding(self.clean_text(sms_text))
            prediction = self.fraud_sms_model.predict(embedding.reshape(1, -1))
            return prediction[0] == 1
        except Exception as e:
            print(f"⚠️ Error during fraud SMS check: {str(e)}")
            return False

    def check_fraud_url(self, url: str) -> bool:
        try:
            embedding = self.generate_embedding(url.strip().lower())
            prediction = self.fraud_url_model.predict(embedding.reshape(1, -1))
            # 1 = legit, so prediction[0] == 0 means phishing
            return prediction[0] == 0
        except Exception as e:
            print(f"⚠️ Error during fraud URL check: {str(e)}")
            return False

    def fetch_real_time_news(self, query: str, max_results: int = 5) -> dict:
        api_key = os.getenv("GNEWS_API_KEY")
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "token": api_key,
            "lang": "en",
            "max": max_results
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = {}
            for idx, article in enumerate(data.get('articles', []), start=1):
                results[f"article_{idx}"] = {
                    "title": article.get('title'),
                    "publishedAt": article.get('publishedAt'),
                    "source": article.get('source', {}).get('name'),
                    "description": article.get('description')
                }

            return results

        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return {}

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def generate_embedding(self, text: str) -> numpy.ndarray:
        try:
            embedding = self.embedder.encode(
                text, batch_size=64, show_progress_bar=False)
            return embedding
        except Exception as e:
            print(f"⚠️ Error generating embedding: {str(e)}")
            return numpy.array([])


if __name__ == '__main__':
    pass
