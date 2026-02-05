from sklearn.ensemble import RandomForestClassifier
import joblib

class VoiceClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        probs = self.model.predict_proba([features])[0]
        pred = self.model.predict([features])[0]
        return pred, probs[pred]

    def save(self, path="voice_model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="voice_model.pkl"):
        self.model = joblib.load(path)
