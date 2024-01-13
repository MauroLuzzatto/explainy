import os


def create_folder(path: str) -> str:
    """Create folder, if it doesn't already exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path
