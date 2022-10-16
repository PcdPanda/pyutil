import importlib


def try_import(module_name: str):
    """Try to import the module given by the module name,
    so that the code can still work with minimum dependency
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    else:
        raise
