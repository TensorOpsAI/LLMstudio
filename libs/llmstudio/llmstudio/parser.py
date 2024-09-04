import json

from pydantic import BaseModel, ValidationError


class Parser:
    def parse_response(self, json_str: str, response_model: BaseModel):
        try:
            start_index = json_str.find("{")
            end_index = json_str.rfind("}")
            trimmed_str = json_str[start_index : end_index + 1]
            if start_index == -1 or end_index == -1:
                raise ValueError("Failed to find JSON in the response.")
            parsed_response = response_model.model_validate(json.loads(trimmed_str))
            return parsed_response
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to decode JSON string: {str(e)}. JSON string: {trimmed_str}\n"
            )
        except Exception as e:
            return e
