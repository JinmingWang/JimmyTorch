import torch

def loadModels(path: str, **models: torch.nn.Module) -> None:
    """
    Load models from a file, handling mismatched, missing, and unexpected parameters gracefully.
    :param path: The path to the file
    :param models: The models to load
    """
    state_dicts = torch.load(path)
    for name, model in models.items():
        print(f"Loading model: {name}")

        if name not in state_dicts:
            print(f"Warning: No state_dict found for model '{name}' in the file. Skipping.")
            continue

        state_dict = state_dicts[name]
        current_state_dict = model.state_dict()

        # Filter and handle mismatched parameters
        mis_matched_keys = set()
        loadable_state_dict = {}
        for param_name, param_value in state_dict.items():
            if param_name in current_state_dict:
                if current_state_dict[param_name].size() == param_value.size():
                    loadable_state_dict[param_name] = param_value
                else:
                    mis_matched_keys.add(param_name)
                    print(f"Warning! Parameter '{param_name}' expect size {current_state_dict[param_name].shape} but got {param_value.shape}. Skipping.")
            else:
                print(f"Unexpected parameter '{param_name}''. Skipping.")

        # Load filtered parameters
        model.load_state_dict(loadable_state_dict, strict=False)

        # Check for missing parameters
        for param_name in current_state_dict.keys():
            if param_name not in loadable_state_dict and param_name not in mis_matched_keys:
                print(f"Missing parameter '{param_name}' in model '{name}'.")


def saveModels(path: str, **models: torch.nn.Module) -> None:
    """
    Save models to a file
    :param path: The path to the file
    :param models: The models to save
    :return: None
    """
    state_dicts = {}
    for name, model in models.items():
        state_dicts[name] = model.state_dict()
    torch.save(state_dicts, path)
