import json
import re

def parse_text(text: str) -> str:
    """
    Converts escaped newline characters (\n) in a string into actual new lines.

    :param text: The input string containing escaped \n characters.
    :return: A formatted string with actual new lines.
    """
    return text.replace("\\n", "\n")



def parse_json_from_string(json_string: str):
    """
    Cleans a JSON string by removing code block markers and parses it into a Python dictionary.

    :param json_string: The JSON string with optional markdown code block markers.
    :return: Parsed JSON as a Python dictionary.
    """
    # Remove code block markers (e.g., ```json ... ```)
    clean_json_string = re.sub(r'^```json\n|\n```$', '', json_string.strip())

    # Parse JSON string into a dictionary
    try:
        return json.loads(clean_json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None  # Return None if parsing fails