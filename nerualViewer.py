import torch
import torch.nn as nn
import joblib

# 1. Load the model and vectorizer
model = nn.Sequential(
    nn.Linear(2000, 128), nn.ReLU(),
    nn.Linear(128, 1), nn.Sigmoid()
)
model.load_state_dict(torch.load('spam_model.pth', map_location='cpu'))
model.eval()

vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict(text):
    # 2. Vectorize the text (dense array -> tensor)
    X = vectorizer.transform([text]).toarray().astype('float32')
    x_tensor = torch.tensor(X)
    
    # 3. Predict
    with torch.no_grad():
        prob = model(x_tensor).item()
        
    # 4. Interpret result
    label = "SPAM" if prob > 0.5 else "HAM"
    print(f"Text: {text}")
    print(f"Spam probability: {prob:.4f}")
    print(f"Prediction: {label}\n")

# --- Test it ---
predict("I won a free coffee at the office today, want to grab it?")
predict("Hey, are we still on for lunch tomorrow?")
