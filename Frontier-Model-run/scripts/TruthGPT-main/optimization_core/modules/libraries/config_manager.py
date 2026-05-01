"""
Configuration Manager
"""

from .imports import *

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        if isinstance(config_source, dict):
            self.config_path = None
            self.config = config_source
        else:
            self.config_path = Path(config_source)
            self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path:
            return {}
            
        if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        elif self.config_path.suffix == '.json':
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Try to load as yaml by default if no extension matches
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        self.config[key] = value
    
    def save(self):
        """Save configuration to file"""
        if not self.config_path:
            # Cannot save if no path provided
            return
            
        if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif self.config_path.suffix == '.json':
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

