import os
from box.exceptions import BoxValueError
import yaml
from XGBClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import datetime
from plotly.io import write_image


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict, save_path=None):
    """Save JSON data with a timestamp in the filename

    Args:
        path (Path): base path to json file
        data (dict): data to be saved in json file
        save_path (Path, optional): Directory path to save the file in. Defaults to None.
    """
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        full_path = save_path / f"{path.stem}_{timestamp}{path.suffix}"
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        full_path = path.parent / f"{path.stem}_{timestamp}{path.suffix}"
    
    with open(full_path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {full_path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"




@ensure_annotations
def save_figure_with_timestamp(fig, prefix="figure", save_path=None):
    '''Saves the given plotly figure with a timestamped filename.'''
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    
    # Determine the full path to save the figure
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        filepath = save_path / filename
    else:
        filepath = Path(filename)
    
    # Save the figure
    write_image(fig, str(filepath))
    
    logger.info(f"Figure saved as: {filepath}")