from pathlib import Path
from typing import Dict, Iterable, Union, BinaryIO, TextIO, Any
from ruamel.yaml import YAML


def parse_multimodal_config_vars(conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a dict of multimodal config values into concrete types.
    """
    out: Dict[str, Any] = {}
    for k, v in conf.items():
        if isinstance(v, list):
            # Convert numeric strings in lists to int
            out[k] = [int(item) if isinstance(item, str) and item.isdigit() else item for item in v]
        elif isinstance(v, str) and v.startswith("0x"):
            out[k] = bytes.fromhex(v[2:])
        elif isinstance(v, str) and v.lower() in ["true", "false"]:
            out[k] = v.lower() == "true"
        elif k not in {"CONFIG_NAME", "MODEL_NAME"} and isinstance(v, str) and v.isdigit():
            out[k] = int(v)
        else:
            out[k] = v
    return out


def load_multimodal_preset(preset_files: Iterable[Union[Path, BinaryIO, TextIO]]) -> Dict[str, Any]:
    """
    Loads and merges multiple multimodal YAML preset config files into one dictionary.
    """
    preset: Dict[str, Any] = {}
    yaml = YAML(typ="base")

    for file in preset_files:
        file_data = yaml.load(file)
        if file_data is None:
            continue
        duplicates = set(file_data.keys()).intersection(preset.keys())
        if duplicates:
            raise ValueError(f"Duplicate config keys found in presets: {', '.join(duplicates)}")
        preset.update(file_data)

    if not preset:
        raise ValueError("No valid config found in preset files.")

    return parse_multimodal_config_vars(preset)


def load_multimodal_config_file(config_path: Union[Path, BinaryIO, TextIO]) -> Dict[str, Any]:
    """
    Loads a single multimodal config YAML file.
    """
    yaml = YAML(typ="base")
    data = yaml.load(config_path)
    return parse_multimodal_config_vars(data or {})


# Global defaults (can be imported elsewhere)
multimodal_base_config: Dict[str, Any]
multimodal_advanced_config: Dict[str, Any]
defaults_loaded = False


def load_multimodal_defaults(configs_path: Path) -> None:
    global multimodal_base_config, multimodal_advanced_config, defaults_loaded

    multimodal_base_config = load_multimodal_config_file(configs_path / "base.yaml")
    multimodal_advanced_config = load_multimodal_config_file(configs_path / "advanced.yaml")

    defaults_loaded = True
