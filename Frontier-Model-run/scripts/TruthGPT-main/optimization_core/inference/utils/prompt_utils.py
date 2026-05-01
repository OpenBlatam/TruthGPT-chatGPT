"""
Prompt utilities for inference engines.

Provides utilities for normalizing and handling prompts consistently
across different inference engine implementations.
"""
import logging
from typing import List, Union, Tuple, Any

logger = logging.getLogger(__name__)


def normalize_prompts(
    prompts: Union[str, List[str]]
) -> Tuple[List[str], bool]:
    """
    Normalize prompts to list format.
    
    Args:
        prompts: Single prompt or list of prompts
    
    Returns:
        Tuple of (prompts_list, was_single)
    
    Raises:
        ValueError: If prompts is empty or invalid
    """
    if isinstance(prompts, str):
        if not prompts.strip():
            raise ValueError("prompt cannot be empty or whitespace")
        return [prompts], True
    
    if isinstance(prompts, list):
        if not prompts:
            raise ValueError("prompts list cannot be empty")
        
        # Validate all prompts are strings and non-empty
        normalized = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise ValueError(
                    f"prompt at index {i} must be a string, got {type(prompt)}"
                )
            if not prompt.strip():
                logger.warning(f"Empty prompt at index {i}, skipping")
                continue
            normalized.append(prompt)
        
        if not normalized:
            raise ValueError("All prompts are empty after normalization")
        
        return normalized, False
    
    raise ValueError(
        f"prompts must be str or List[str], got {type(prompts)}"
    )


def handle_single_prompt(
    results: List[str],
    was_single: bool
) -> Union[str, List[str]]:
    """
    Convert results back to single value if original was single prompt.
    
    Args:
        results: List of results
        was_single: Whether original input was a single prompt
    
    Returns:
        Single result if was_single, otherwise list
    """
    if was_single:
        return results[0] if results else ""
    return results


def extract_generated_text(
    outputs: List[Any],
    output_attr: str = "text",
    fallback: str = ""
) -> List[str]:
    """
    Extract generated text from output objects.
    
    Args:
        outputs: List of output objects
        output_attr: Attribute name to extract (e.g., "text", "outputs[0].text")
        fallback: Fallback value if extraction fails
    
    Returns:
        List of generated texts
    """
    results = []
    
    for i, output in enumerate(outputs):
        try:
            # Handle nested attributes like "outputs[0].text"
            if "[" in output_attr and "]" in output_attr:
                # Parse "outputs[0].text"
                parts = output_attr.split("[")
                attr_name = parts[0]
                index_part = parts[1].split("]")[0]
                index = int(index_part)
                sub_attr = parts[1].split("]")[1].lstrip(".")
                
                obj = getattr(output, attr_name, None)
                if obj and isinstance(obj, (list, tuple)) and len(obj) > index:
                    text = getattr(obj[index], sub_attr, fallback)
                    results.append(text if text else fallback)
                else:
                    logger.warning(f"Could not extract text from output {i}")
                    results.append(fallback)
            else:
                # Simple attribute access
                text = getattr(output, output_attr, fallback)
                results.append(text if text else fallback)
                
        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(
                f"Failed to extract text from output {i}: {e}. Using fallback."
            )
            results.append(fallback)
    
    return results


def truncate_prompts(
    prompts: List[str],
    max_batch_size: int
) -> List[str]:
    """
    Truncate prompts list to max batch size.
    
    Args:
        prompts: List of prompts
        max_batch_size: Maximum batch size
    
    Returns:
        Truncated list of prompts
    """
    if len(prompts) <= max_batch_size:
        return prompts
    
    logger.warning(
        f"Truncating {len(prompts)} prompts to {max_batch_size} "
        f"(max_batch_size limit)"
    )
    return prompts[:max_batch_size]













