Wizard Mode = Parameter Graph + Dynamic Execution

class MultimodalInput:
    text: str
    audio: Optional[bytes]  # base64 or raw
    image: Optional[bytes]
    context: Optional[Dict[str, Any]]


MultimodalInput encapsulates different types of user inputs (text, audio, image, and context).

ActionGraph manages the dynamic execution of tasks based on parsed input.

Actions like send_message, send_email, and open_camera are predefined functions that can be dynamically added and executed based on user input.

The parse_input function interprets the user’s textual input and generates the corresponding action and parameters.

