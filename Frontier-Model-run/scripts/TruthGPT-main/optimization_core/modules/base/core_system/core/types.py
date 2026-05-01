"""
Common type definitions and type aliases for optimization_core.

Provides reusable type hints to ensure consistency across all modules.
"""
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Any,
    Callable,
    Tuple,
    Set,
    TypeVar,
    Generic,
    Protocol,
    runtime_checkable,
)
from pathlib import Path
from enum import Enum

# ════════════════════════════════════════════════════════════════════════════════
# TYPE ALIASES
# ════════════════════════════════════════════════════════════════════════════════

# Path types
PathLike = Union[str, Path]

# Numeric types
Number = Union[int, float]
PositiveInt = int  # Should be validated to be > 0
NonNegativeInt = int  # Should be validated to be >= 0
PositiveFloat = float  # Should be validated to be > 0.0
NonNegativeFloat = float  # Should be validated to be >= 0.0

# Collection types
StringList = List[str]
IntList = List[int]
FloatList = List[float]
DictStrAny = Dict[str, Any]
DictStrStr = Dict[str, str]
DictStrInt = Dict[str, int]
DictStrFloat = Dict[str, float]

# Optional types
OptionalStr = Optional[str]
OptionalInt = Optional[int]
OptionalFloat = Optional[float]
OptionalPath = Optional[PathLike]
OptionalDict = Optional[DictStrAny]
OptionalList = Optional[List[Any]]

# Callable types
CallableNoArgs = Callable[[], Any]
CallableOneArg = Callable[[Any], Any]
CallableTwoArgs = Callable[[Any, Any], Any]

# ════════════════════════════════════════════════════════════════════════════════
# GENERIC TYPE VARIABLES
# ════════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
R = TypeVar('R')  # Return type

# ════════════════════════════════════════════════════════════════════════════════
# PROTOCOLS
# ════════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to dict."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create object from dictionary."""
        ...


@runtime_checkable
class Validatable(Protocol):
    """Protocol for objects that can be validated."""
    
    def validate(self) -> None:
        """Validate object state."""
        ...


@runtime_checkable
class Resettable(Protocol):
    """Protocol for objects that can be reset."""
    
    def reset(self) -> None:
        """Reset object to initial state."""
        ...


# ════════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════════

class Precision(str, Enum):
    """Data precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    AUTO = "auto"


class QuantizationType(str, Enum):
    """Quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"


class BackendType(str, Enum):
    """Backend types for polyglot architecture."""
    AUTO = "auto"
    RUST = "rust"
    CPP = "cpp"
    GO = "go"
    PYTHON = "python"
    JULIA = "julia"
    ELIXIR = "elixir"
    SCALA = "scala"


class DeviceType(str, Enum):
    """Device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


__all__ = [
    # Type aliases
    "PathLike",
    "Number",
    "PositiveInt",
    "NonNegativeInt",
    "PositiveFloat",
    "NonNegativeFloat",
    "StringList",
    "IntList",
    "FloatList",
    "DictStrAny",
    "DictStrStr",
    "DictStrInt",
    "DictStrFloat",
    "OptionalStr",
    "OptionalInt",
    "OptionalFloat",
    "OptionalPath",
    "OptionalDict",
    "OptionalList",
    "CallableNoArgs",
    "CallableOneArg",
    "CallableTwoArgs",
    # Type variables
    "T",
    "K",
    "V",
    "R",
    # Protocols
    "Serializable",
    "Validatable",
    "Resettable",
    # Enums
    "Precision",
    "QuantizationType",
    "BackendType",
    "DeviceType",
]













