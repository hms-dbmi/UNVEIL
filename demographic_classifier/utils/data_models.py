from enum import Enum
import torch
from pydantic import BaseModel
from metrics import MetricManager
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class AutoEnum(Enum):
    """Enum wrapper which avoid the need to call .value() """

    def __call__(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class DATASET_TYPES(AutoEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    EXTERNAL_VAL = "external_val"
    EVAL = 'eval'


class PTBundle(BaseModel):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None = None

    # Allows Pydantic to handle arbitrary types like nn.Module and Optimizer
    class Config:
        arbitrary_types_allowed = True  

    @classmethod
    def load_bundle(cls, load_path: str, map_location: str) -> 'PTBundle':
        # Load the bundle dictionary from the saved file
        bundle_dict = torch.load(load_path, map_location=map_location, weights_only=False)

        # Initialize PTBundle using the loaded dictionary
        return cls(**bundle_dict)

    def save_bundle(self, save_path: str) -> None:
        # Convert the pydantic model to a dictionary
        bundle_dict = self.model_dump()
        # Save the dictionary using torch.save
        torch.save(bundle_dict, save_path)