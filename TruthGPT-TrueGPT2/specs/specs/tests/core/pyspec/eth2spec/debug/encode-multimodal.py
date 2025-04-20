import base64
from typing import Union as TypingUnion

# Base multimodal datatypes
class Text:
    def __init__(self, content: str):
        self.content = content

class Image:
    def __init__(self, data: bytes, format: str = "jpeg"):
        self.data = data
        self.format = format

class Audio:
    def __init__(self, data: bytes, format: str = "mp3"):
        self.data = data
        self.format = format

class Video:
    def __init__(self, data: bytes, format: str = "mp4"):
        self.data = data
        self.format = format

class MultimodalContainer:
    def __init__(self, **kwargs):
        self.fields = kwargs

    def items(self):
        return self.fields.items()

# Encoder function
def encode_multimodal(value, include_metadata=False):
    if isinstance(value, Text):
        return {
            "type": "text",
            "content": value.content,
        }
    elif isinstance(value, Image):
        return {
            "type": "image",
            "format": value.format,
            "data": base64.b64encode(value.data).decode(),
        }
    elif isinstance(value, Audio):
        return {
            "type": "audio",
            "format": value.format,
            "data": base64.b64encode(value.data).decode(),
        }
    elif isinstance(value, Video):
        return {
            "type": "video",
            "format": value.format,
            "data": base64.b64encode(value.data).decode(),
        }
    elif isinstance(value, MultimodalContainer):
        result = {}
        for key, val in value.items():
            result[key] = encode_multimodal(val, include_metadata)
        if include_metadata:
            result["_metadata"] = {"fields_count": len(value.fields)}
        return result
    elif isinstance(value, list):
        return [encode_multimodal(v, include_metadata) for v in value]
    elif isinstance(value, dict):
        return {k: encode_multimodal(v, include_metadata) for k, v in value.items()}
    else:
        raise Exception(f"Unknown multimodal value: {value}, type={type(value)}")
