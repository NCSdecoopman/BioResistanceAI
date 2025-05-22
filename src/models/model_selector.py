import yaml, importlib

def load_models(yaml_path: str):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    models = {}
    for name, info in config["models"].items():
        module, class_name = info["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module), class_name)
        models[name] = {"class": cls, "params": info["params"]}
    return models
