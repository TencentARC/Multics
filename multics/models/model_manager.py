import os, torch
from .registry import model_loader_configs
from .utils import (
    load_state_dict,
    init_weights_on_device,
    hash_state_dict_keys,
    split_state_dict_with_prefix,
)


def load_model_from_single_file(
    state_dict, model_names, model_classes, model_resource, torch_dtype, device
):
    loaded_model_names, loaded_models = ([], [])
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(
                f"        This model is initialized with extra kwargs: {extra_kwargs}"
            )
        else:
            model_state_dict, extra_kwargs = (state_dict_results, {})
        torch_dtype = (
            torch.float32
            if extra_kwargs.get("upcast_to_float32", False)
            else torch_dtype
        )
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()
        model.load_state_dict(model_state_dict, assign=True)
        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return (loaded_model_names, loaded_models)


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(
        self,
        keys_hash,
        keys_hash_with_shape,
        model_names,
        model_classes,
        model_resource,
    ):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (
            model_names,
            model_classes,
            model_resource,
        )
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (
                model_names,
                model_classes,
                model_resource,
            )

    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True
        return False

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        **kwargs,
    ):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[
                keys_hash_with_shape
            ]
            loaded_model_names, loaded_models = load_model_from_single_file(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
            )
            return (loaded_model_names, loaded_models)
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            loaded_model_names, loaded_models = load_model_from_single_file(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
            )
            return (loaded_model_names, loaded_models)
        return (loaded_model_names, loaded_models)


class ModelDetectorFromSplitedSingleFile(ModelDetectorFromSingleFile):
    def __init__(self, model_loader_configs=[]):
        super().__init__(model_loader_configs)

    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                return True
        return False

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        **kwargs,
    ):
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        valid_state_dict = {}
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                valid_state_dict.update(sub_state_dict)
        if super().match(file_path, valid_state_dict):
            loaded_model_names, loaded_models = super().load(
                file_path, valid_state_dict, device, torch_dtype
            )
        else:
            loaded_model_names, loaded_models = ([], [])
            for sub_state_dict in splited_state_dict:
                if super().match(file_path, sub_state_dict):
                    loaded_model_names_, loaded_models_ = super().load(
                        file_path, valid_state_dict, device, torch_dtype
                    )
                    loaded_model_names += loaded_model_names_
                    loaded_models += loaded_models_
        return (loaded_model_names, loaded_models)


class ModelManager:
    def __init__(self, torch_dtype=torch.float16, device="cuda"):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        self.model_detector = [
            ModelDetectorFromSingleFile(model_loader_configs),
            ModelDetectorFromSplitedSingleFile(model_loader_configs),
        ]

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype
        _fp0 = file_path[0] if isinstance(file_path, list) and file_path else file_path
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            state_dict = None
        if isinstance(file_path, list):
            _branch = "list"
        elif isinstance(file_path, str) and os.path.isfile(file_path):
            _branch = "file"
        else:
            _branch = "none"
        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                model_names, models = model_detector.load(
                    file_path,
                    state_dict,
                    device=device,
                    torch_dtype=torch_dtype,
                    allowed_model_names=model_names,
                    model_manager=self,
                )
                for model_name, model in zip(model_names, models):
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(model_name)
                print(f"    The following models are loaded: {model_names}.")
                break
        else:
            print(f"    We cannot detect the model type. No models are loaded.")

    def load_models(
        self, file_path_list, model_names=None, device=None, torch_dtype=None
    ):
        for file_path in file_path_list:
            self.load_model(
                file_path, model_names, device=device, torch_dtype=torch_dtype
            )

    def fetch_model(
        self, model_name, file_path=None, require_model_path=False, index=None
    ):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(
            self.model, self.model_path, self.model_name
        ):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        elif index is None:
            model = fetched_models[0]
            path = fetched_model_paths[0]
            print(
                f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[0]}."
            )
        elif isinstance(index, int):
            model = fetched_models[:index]
            path = fetched_model_paths[:index]
            print(
                f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[:index]}."
            )
        else:
            model = fetched_models
            path = fetched_model_paths
            print(
                f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths}."
            )
        if require_model_path:
            return (model, path)
        else:
            return model

    def to(self, device):
        for model in self.model:
            model.to(device)
