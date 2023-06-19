import os
from huggingface_hub import hf_api

HOME_PATH = os.path.expanduser('~')
CACHE_PATH = os.path.join(HOME_PATH, './cache/trk/')
FILE_PATH = 'current_org'

def set_org(org: str):
    file_path = os.path.join(CACHE_PATH, FILE_PATH)

    # Create the directory if it does not exist
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    # Create a new text file in the directory
    with open(file_path, 'w') as file:
        file.write(org)  # write an empty string to the file


def get_org() -> str:
    file_path = os.path.join(CACHE_PATH, FILE_PATH)

    if not os.path.isfile(file_path):
        raise ValueError(f"{file_path} does not exist. Make sure to run `trk.setup_repo(...)` first.")
    with open(file_path, 'r') as file:
        current_org = file.read()

    return current_org
