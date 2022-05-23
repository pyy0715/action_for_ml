from typing import List

from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.types import JsonSerializable


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact("sklearn_model")])
class ModelApiService(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        input_data = parsed_jsons[0]["data"]
        pred_y = self.artifacts.model.predict(input_data)
        return pred_y
