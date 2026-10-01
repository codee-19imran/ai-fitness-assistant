import joblib

# Load model
model = joblib.load("talent_model.pkl")

# Example input: [speed, stride_length, stamina, height]
sample = [[185, 80, 24, 90, 88, 85, 92, 90, 90, 92, 88]]

prediction = model.predict(sample)

print("Predicted Talent Level:", prediction[0])
