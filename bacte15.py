from diz_predictor import DIZPredictor

# Load the predictor
predictor = DIZPredictor(r"C:\Users\Amare\Desktop\pysr_final_manual_save\scaler.pkl")

# Make a prediction
diz = predictor.predict(Cbac=10000000, CAgNP=500, Ra=30)
print(f"Predicted DIZ: {diz:.2f}")