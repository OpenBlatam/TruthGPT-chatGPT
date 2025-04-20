def decode_multimodal(data: Any, typ: type):
    if isinstance(typ, type) and issubclass(typ, list):
        # List with element type provided
        elem_type = typ.__args__[0]
        return [decode_multimodal(item, elem_type) for item in data]

    if typ == Text:
        return Text(content=data["content"])

    elif typ == Image:
        return Image(
            data=base64.b64decode(data["data"]),
            format=data.get("format", "jpeg")
        )

    elif typ == Audio:
        return Audio(
            data=base64.b64decode(data["data"]),
            format=data.get("format", "mp3")
        )

    elif typ == Video:
        return Video(
            data=base64.b64decode(data["data"]),
            format=data.get("format", "mp4")
        )

    elif issubclass(typ, MultimodalContainer):
        fields = typ.fields()
        temp = {}
        for field_name, field_type in fields.items():
            temp[field_name] = decode_multimodal(data[field_name], field_type)
        return typ(**temp)

    else:
        raise Exception(f"Type not recognized: data={data}, typ={typ}")
