# TODO AUG 06 version (safe model loader)
"""
This package wires up model discovery/creation.

Add '<name>_model.py' under models/ and define a class '<CamelName>Model' that
inherits BaseModel. Invoke with '--model <name>'.
"""

import importlib
import inspect
from typing import Type
from models.base_model import BaseModel


def find_model_using_name(model_name: str) -> Type[BaseModel]:
    """
    Import 'models/<model_name>_model.py' and return the model class.

    We first try to match a class whose name (lowercased) equals:
        model_name.replace('_','') + 'model'
    If not found, we fallback to the ONLY BaseModel subclass in the module.
    """
    module_name = f"models.{model_name}_model"
    try:
        modellib = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(
            f"Could not import module '{module_name}'. "
            f"Make sure 'models/{model_name}_model.py' exists and is importable."
        ) from e

    target = model_name.replace('_', '') + 'model'  # expected lowercase class name

    candidates = []
    selected = None
    for name, obj in vars(modellib).items():
        # only consider classes
        if not inspect.isclass(obj):
            continue
        # must subclass BaseModel (but not BaseModel itself)
        if not issubclass(obj, BaseModel) or obj is BaseModel:
            continue

        candidates.append(obj)
        if name.lower() == target:
            selected = obj

    if selected is not None:
        return selected

    if len(candidates) == 1:
        # fallback: exactly one BaseModel subclass in the file
        return candidates[0]

    # build a helpful message
    available = ", ".join([c.__name__ for c in candidates]) or "none"
    raise RuntimeError(
        f"In {module_name}.py, could not find a subclass of BaseModel with class name "
        f"matching '{target}'. Available BaseModel subclasses: {available}. "
        f"Ensure your class is named like '<CamelName>Model' and your file is named '{model_name}_model.py'."
    )


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt) -> BaseModel:
    """
    Create a model instance given the parsed options.

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model_class = find_model_using_name(opt.model)
    instance = model_class(opt)
    print(f"model [{type(instance).__name__}] was created")
    return instance
