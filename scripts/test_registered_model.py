import os
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USER"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

print("Loading registered model...")
model = mlflow.pyfunc.load_model(
    "models:/ScientificClaimVerifier/1"
)

claims = pd.DataFrame({
    "claim": [
        "Vitamin D supplementation reduces depression."
    ]
})

print("Running verification...")
result = model.predict(claims)

print(result.to_string(index=False))
print("\nRegistered model inference successful!")