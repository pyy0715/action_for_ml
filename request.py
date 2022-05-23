import requests

from model_utils import load_model
from service import ModelApiService

if __name__ == "__main__":
    # Create a service instance
    service = ModelApiService()
    model = load_model(model_name="ElasticnetWineModel", model_version=1)

    # Pack the newly trained model artifact
    service.pack("sklearn_model", model)

    # Save the prediction service to disk for model serving
    saved_path = service.save()

    headers = {"Content-Type": "application/json"}
    json_data = {"data": [[3, 0, 1, 1, 2], [2, 3, 2, 2, 2]]}

    response = requests.post(
        "http://127.0.0.1:51437/predict", headers=headers, json=json_data
    )
    print(response.text)
