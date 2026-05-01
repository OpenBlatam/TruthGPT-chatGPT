"""
Core components for Advanced Modular Library System
"""

from .imports import *

class BaseModule(ABC):
    """Base class for all modular components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Setup the module"""
        pass
    
    def forward(self, *args, **kwargs):
        """Forward pass (optional)"""
        pass
    
    def to(self, device):
        """Move module to device"""
        self.device = device
        return self
    
    def save(self, path: str):
        """Save module state"""
        # Check if attribute exists before saving
        if hasattr(self, 'state_dict'):
             torch.save(self.state_dict(), path)
        else:
             self.logger.warning(f"Module {self.__class__.__name__} has no state_dict to save")
    
    def load(self, path: str):
        """Load module state"""
        if hasattr(self, 'load_state_dict'):
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.logger.warning(f"Module {self.__class__.__name__} has no load_state_dict method")

