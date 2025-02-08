from pathlib import Path
from typing import Any

import safetensors.torch as st
import timm
import torch
import torch.nn as nn


class BackboneNotFoundException(Exception):
    pass


def list_available() -> list[str]:
    models = torch.hub.list("pytorch/vision")
    try:
        import timm

        models += timm.list_models()
    except Exception:
        pass

    return set(models)


def num_params(model: "nn.Module", only_trainable=False):
    return sum(
        [p.numel() for p in model.parameters() if not only_trainable or p.requires_grad]
    )


class TimmModule(nn.Module):
    def __init__(
        self, model: nn.Module, feature_info: timm.models._features.FeatureInfo
    ):
        super().__init__()
        self.model = model
        self.feature_info = feature_info

        self.num_params = num_params(self)

        self.feature_dims: dict[str, torch.Size] | None = None
        self.last_dim: torch.Size = None

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


class TimmFeatureExtractor(TimmModule):
    def __init__(
        self,
        model: nn.Module,
        out_indices: list[int],
        image_size: tuple[int, int, int],
    ):
        super().__init__(model=model, feature_info=model.feature_info.info)
        self.layers = [layer["module"] for layer in self.feature_info]

        o: torch.Tensor = model(torch.randn(1, *image_size))
        self.feature_dims = {self.layers[i]: x.shape for i, x in zip(out_indices, o)}
        self.last_dim = self.feature_dims[self.layers[out_indices[-1]]]

    @property
    def features(self):
        return self.feature_dims.keys()

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        return dict(zip(self.layers, x))


class TimmBackbone(TimmModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: tuple[int, int, int],
    ):
        super().__init__(model=model, feature_info=model.feature_info)

        o: torch.Tensor = model(torch.randn(1, *image_size))
        self.last_dim = o[0].shape


class Backbone:
    """
    Class that allows to import models from timm and load pretrained checkpoints from a local file or remotely
    This class also allows to convert models to feature extractors by providing the layers parameter in .load

    Note: Stored in a class to allow for overwriting methods easily
    """

    @staticmethod
    def _load_torch_cp(model: "nn.Module", path: Path):
        cp = torch.load(str(path), map_location=torch.device("cpu"), weights_only=True)
        if "state_dict" in cp:
            cp = cp["state_dict"]

        # From timm
        if "model" in cp:
            cp: dict[str, Any] = cp["model"]
            cp_ = {}
            for k, v in cp.items():
                cp_[k.replace("module.", "")] = v
            cp = cp_

        model.load_state_dict(cp)

    @staticmethod
    def _load_pretrained(
        model: "nn.Module",
        name: Path | str,
        checkpoint_folder: str | Path = None,
        checkpoint_path: str | Path = None,
        use_safetensors: bool = True,
    ):
        if checkpoint_path is not None:
            cp_path = checkpoint_path
        else:
            cp_folder = Path(checkpoint_folder or ".")
            cp_path = (
                cp_folder / f"{name}.{'safetensors' if use_safetensors else 'pth'}"
            )

        print(f"Loading weights at {cp_path}")

        if use_safetensors:
            st.load_model(model, cp_path)
        else:
            Backbone._load_torch_cp(model, cp_path)

    @staticmethod
    def load(
        name: str,
        num_classes=0,
        global_pool=None,
        layers: list[int] | None = None,
        pre_trained: bool = True,
        checkpoint_folder: str | Path = None,
        checkpoint_path: str | Path = None,
        use_safetensors: bool = True,
        image_size: tuple[int, int, int] | None = (3, 256, 256),
        cache_dir: str | None = None,
        **kwargs,
    ):
        """
        Args:
            name (str): Model's name
            layers (list[str] | None, optional): If you wish to load the model as a feature extractor, provide the model layers to extract features from. Defaults to None.
            pre_trained (bool, optional): Load pretrained model checkpoint. Defaults to True.
            checkpoint_folder (str | Path, optional): Loading model checkpoint locally from a given folder - will use the model name for the checkpoint filename. Defaults to None.
            checkpoint_path (str | Path, optional): Loading model checkpoint locally from a given exact path - has priority over checkpoint_folder is both are passed. Defaults to None.
            use_safetensors (bool, optional): Load the model checkpoint from a safetensors file. Defaults to True.
            with_dims (bool, optional): Compute intermediate dimensions given an input of image_size (see image_size parameter). Defaults to True.
            use_timm (bool, optional): Whether to load the model from timm and not torchvision - useful when both torchvision and timm provide the same model. Defaults to False.
            image_size (tuple[int, int, int], optional): The input image size to compute intermediate dimensions from (see with_dims parameter). Defaults to (3, 256, 256).

        Returns:
            nn.Module: Loaded backbone module
        """
        remote_checkpoint: bool = (
            pre_trained
            if checkpoint_folder is None and checkpoint_path is None
            else False
        )

        model: nn.Module = timm.create_model(
            model_name=name,
            # Don't use remote checkpoint if a local checkpoints folder or path is passed
            pretrained=remote_checkpoint,
            num_classes=num_classes,
            global_pool=global_pool,
            features_only=True if layers is not None else False,
            out_indices=layers,
            cache_dir=cache_dir,
            **kwargs,
        )

        if pre_trained and not remote_checkpoint:
            Backbone._load_pretrained(
                model=model,
                name=name,
                checkpoint_folder=checkpoint_folder,
                checkpoint_path=checkpoint_path,
                use_safetensors=use_safetensors,
            )

        image_size = image_size or (3, 224, 224)
        if layers is not None:
            model = TimmFeatureExtractor(
                model=model, out_indices=layers, image_size=image_size
            )
        else:
            model = TimmBackbone(model=model, image_size=image_size)

        print(f"Backbone has {model.num_params:,} params")

        return model
