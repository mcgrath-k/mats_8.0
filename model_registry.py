# model_registry.py

import os
import yaml
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Union
import torch
import pandas as pd
from transformer_lens import HookedTransformer, HookedTransformerConfig

@dataclass
class ModelConfig:
    # Model configuration parameters
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    n_ctx: int
    d_vocab: int
    act_fn: str = "gelu"
    attention_dir: str = "causal"
    attn_only: bool = False
    use_attn_result: bool = True
    normalization_type: Optional[str] = None
    positional_embedding_type: str = "shortformer"
    tokenizer_name: Optional[str] = None
    seed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        # Only include keys that are in the dataclass annotations.
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict:
        return asdict(self)

class ModelRegistry:
    """
    Registry for saving, loading, and listing models.
    
    Each model is stored in its own folder inside `models_dir` (default "models"). 
    A valid model folder must contain two files:
      - model_config.yaml : the YAML file representing the ModelConfig.
      - model_weights.pt  : the PyTorch model state dictionary.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def get_model_folder(self, name: str) -> str:
        """Return the full folder path for a given model name."""
        return os.path.join(self.models_dir, name)

    def save_model(self, name: str, model: torch.nn.Module, config: ModelConfig) -> None:
        """
        Save the model weights and its configuration under a folder named `name`.
        The configuration is saved as YAML (model_config.yaml) and the weights as model_weights.pt.
        """
        folder = self.get_model_folder(name)
        os.makedirs(folder, exist_ok=True)
        config_path = os.path.join(folder, "model_config.yaml")
        weights_path = os.path.join(folder, "model_weights.pt")
        with open(config_path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        torch.save(model.state_dict(), weights_path)
        print(f"Model saved in folder: {folder}")

    def load_model(self, name: str) -> Tuple[HookedTransformer, ModelConfig]:
        """
        Load the model and configuration for the given model name.
        Returns a tuple of (model, config). The model is instantiated using HookedTransformerConfig.
        """
        folder = self.get_model_folder(name)
        config_path = os.path.join(folder, "model_config.yaml")
        weights_path = os.path.join(folder, "model_weights.pt")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Folder '{folder}' must contain both model_config.yaml and model_weights.pt")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = ModelConfig.from_dict(config_dict)
        # Build a HookedTransformerConfig from ModelConfig (only matching fields)
        ht_config_kwargs = {k: v for k, v in config.to_dict().items() if k in HookedTransformerConfig.__annotations__}
        ht_config = HookedTransformerConfig(**ht_config_kwargs)
        model = HookedTransformer(ht_config)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model, config

    def list_models(self) -> List[Dict]:
        """
        List all models in the registry.
        Returns a list of dictionaries where each dictionary contains all ModelConfig fields along with a 'model_name' key.
        """
        models = []
        for entry in os.listdir(self.models_dir):
            folder = os.path.join(self.models_dir, entry)
            if os.path.isdir(folder):
                config_path = os.path.join(folder, "model_config.yaml")
                weights_path = os.path.join(folder, "model_weights.pt")
                if os.path.exists(config_path) and os.path.exists(weights_path):
                    with open(config_path, "r") as f:
                        config_dict = yaml.safe_load(f)
                    # Add the model name for identification.
                    config_dict["model_name"] = entry
                    models.append(config_dict)
        return models

    def list_models_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame listing all models with their configuration details.
        """
        models_list = self.list_models()
        return pd.DataFrame(models_list)

# Example usage (if run as main)
if __name__ == "__main__":
    # Create a registry instance (will create "models" folder if not exists)
    registry = ModelRegistry()

    # Define a sample configuration
    sample_config = ModelConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True,  # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b",
        seed=398,
        use_attn_result=True,
        normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer",
    )

    # (Example) Initialize a model using HookedTransformerConfig built from sample_config:
    ht_config_kwargs = {k: v for k, v in sample_config.to_dict().items() if k in HookedTransformerConfig.__annotations__}
    ht_config = HookedTransformerConfig(**ht_config_kwargs)
    model = HookedTransformer(ht_config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Save the model (this example uses the name "arithmetic_model_4layers")
    registry.save_model("arithmetic_model_4layers", model, sample_config)

    # List all models (as DataFrame)
    df = registry.list_models_dataframe()
    print("Registered models:")
    print(df)