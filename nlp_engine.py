import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class NLPSentimentEngine:
    def __init__(self):
        self.pipeline = None
        self._train_model()

    def _train_model(self):
        # A small, clean training corpus representing typical diary and emotional logs.
        # This allows us to train a local ML classifier instantly on startup without downloads or internet access.
        training_data = [
            # Positive
            ("I feel happy and excited today.", "positive"),
            ("Great day today, very productive and fun.", "positive"),
            ("I am feeling wonderful and full of energy.", "positive"),
            ("Loved the team work and we got excellent results.", "positive"),
            ("Success! We finally completed our tasks.", "positive"),
            ("Feeling fantastic and motivated about the future.", "positive"),
            ("It was a joyful and peaceful afternoon.", "positive"),
            ("Extremely satisfied with how things went.", "positive"),
            ("I am feeling happy and hopeful.", "positive"),
            ("Had an amazing workout and delicious food.", "positive"),
            ("So glad that I got to talk with my family.", "positive"),
            ("Today was awesome, achieved all my goals.", "positive"),
            
            # Negative
            ("I feel sad and depressed.", "negative"),
            ("Terrible day, nothing worked out as planned.", "negative"),
            ("I am feeling awful and have a headache.", "negative"),
            ("Frustrated with the slow progress and failures.", "negative"),
            ("Failing to make any headway, feeling stuck.", "negative"),
            ("Feeling exhausted, stressed, and overwhelmed.", "negative"),
            ("Angry and disappointed with the outcome.", "negative"),
            ("Anxious about the deadline and worried about failing.", "negative"),
            ("Very lonely and blue today.", "negative"),
            ("It was a horrible experience, feeling down.", "negative"),
            ("Disgusted by the behaviour, feeling unhappy.", "negative"),
            ("My energy is extremely low, feeling burnt out.", "negative"),
            
            # Neutral
            ("I went for a standard walk today.", "neutral"),
            ("Had lunch at 12:30 PM.", "neutral"),
            ("Completed some documentation and read files.", "neutral"),
            ("Normal day at the office, nothing special.", "neutral"),
            ("Nothing new happened, just standard routine.", "neutral"),
            ("Sitting down and reading a technical book.", "neutral"),
            ("Writing code on my laptop in the room.", "neutral"),
            ("Plain tasks and attended the scheduled meeting.", "neutral"),
            ("The weather is cloudy and mild.", "neutral"),
            ("Cleaned up my desk and sorted some emails.", "neutral"),
            ("Eating an apple and watching the news.", "neutral"),
            ("We discussed the project guidelines briefly.", "neutral")
        ]

        texts, labels = zip(*training_data)
        
        # Build a robust TF-IDF + Logistic Regression pipeline
        # Using sublinear TF scaling to handle varied length text
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ('clf', LogisticRegression(C=1.0, max_iter=200))
        ])
        
        self.pipeline.fit(texts, labels)
        print("NLP Sentiment Engine: Custom ML model trained successfully.")

    def analyze_sentence(self, sentence):
        """
        Analyzes a single sentence.
        Returns:
            polarity: float between -1 (negative) and 1 (positive)
            label: string 'Positive', 'Negative', or 'Neutral'
        """
        if not sentence or not sentence.strip():
            return 0.0, "Neutral"

        # Get probability estimates
        classes = self.pipeline.classes_
        probs = self.pipeline.predict_proba([sentence])[0]
        
        prob_dict = dict(zip(classes, probs))
        
        # Calculate a polarity index: Positive prob minus Negative prob
        pos_prob = prob_dict.get('positive', 0.0)
        neg_prob = prob_dict.get('negative', 0.0)
        neu_prob = prob_dict.get('neutral', 0.0)
        
        polarity = pos_prob - neg_prob
        
        # Determine final label based on max probability
        max_class = max(prob_dict, key=prob_dict.get)
        
        if max_class == 'positive':
            label = "Positive"
        elif max_class == 'negative':
            label = "Negative"
        else:
            # If neutral is dominant or polarity is very close to 0
            if abs(polarity) < 0.15:
                label = "Neutral"
            else:
                label = "Positive" if polarity > 0 else "Negative"

        return float(polarity), label

    def analyze_document(self, text):
        """
        Splits a larger document into sentences, analyzes each,
        and averages the polarity.
        """
        if not text or not text.strip():
            return 0.0, "Neutral"
            
        # Basic sentence splitting (handles periods, exclamation marks, question marks)
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return self.analyze_sentence(text)
            
        polarities = []
        labels_count = {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        for s in sentences:
            pol, lbl = self.analyze_sentence(s)
            polarities.append(pol)
            labels_count[lbl] += 1
            
        avg_polarity = np.mean(polarities)
        
        # Get dominant label
        dominant_label = max(labels_count, key=labels_count.get)
        
        # Adjust label if average polarity is very strong in one direction
        if avg_polarity > 0.35 and dominant_label == "Neutral":
            dominant_label = "Positive"
        elif avg_polarity < -0.35 and dominant_label == "Neutral":
            dominant_label = "Negative"
            
        return float(avg_polarity), dominant_label

# Quick test if run directly
if __name__ == "__main__":
    engine = NLPSentimentEngine()
    test_texts = [
        "Today was a beautiful day! I got so much work done and felt very happy.",
        "I am feeling incredibly tired and stressed about my upcoming project defense. It is exhausting.",
        "I had dinner and read a paper. Now I am going to bed."
    ]
    for t in test_texts:
        pol, lbl = engine.analyze_document(t)
        print(f"Text: {t}\nPolarity: {pol:.2f} | Label: {lbl}\n")
