from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import joblib
import numpy as np
import os

# Load model and scaler using absolute path
model_path = os.path.join(settings.BASE_DIR, "model.pkl")
scaler_path = os.path.join(settings.BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

FEATURE_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]

@api_view(['GET'])
def health_check(request):
    return Response({"status": "API is healthy"}, status=status.HTTP_200_OK)

@api_view(['POST'])
def predict_diabetes(request):
    try:
        input_data = [request.data.get(f) for f in FEATURE_NAMES]
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        return Response({"prediction": int(prediction)})
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def feature_importance(request):
    try:
        importances = model.feature_importances_
        return Response({"feature_importance": dict(zip(FEATURE_NAMES, importances.tolist()))})
    except:
        return Response({"error": "Model doesn't support feature importances"}, status=status.HTTP_400_BAD_REQUEST)
