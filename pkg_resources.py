import importlib.resources

def resource_filename(package_or_requirement, resource_name):
    return str(importlib.resources.files(package_or_requirement).joinpath(resource_name))