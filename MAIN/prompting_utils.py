from pydantic import BaseModel
from model_utils import get_regression_fields, get_optional_regression_fields, unwrap_type, field_is_flag_model
import json




def generate_prompt_schema(model: type[BaseModel]) -> dict:
    """
    Genera lo schema JSON per il prompt, basato sui campi del modello Pydantic.
    """
    reg_fields = get_regression_fields(model)
    optional_reg_fields = get_optional_regression_fields(model)
    result = dict()
    for f in model.model_fields:
        if f in reg_fields:
            field_type_str = "int"
            field_type = unwrap_type(model.model_fields[f].annotation)
            if field_type is float:
                field_type_str = "float"
                
            if f in optional_reg_fields:
                field_type_str = f'{field_type_str} | null'
            result[f] = field_type_str
        elif field_is_flag_model(f, model):
            field_type = unwrap_type(model.model_fields[f].annotation)
            result[f] = generate_prompt_schema(field_type)
        else:
            field_type = unwrap_type(model.model_fields[f].annotation)
            values = [e.value for e in field_type]
            result[f] = " | ".join(values)
    return result


def create_system_prompt(prompt_path: str, annotation_model: type[BaseModel]) -> str:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
        
    schema = json.dumps(generate_prompt_schema(annotation_model), indent=None)
    system_prompt = system_prompt.replace("{schema_json}", schema)
    return system_prompt


if __name__ == "__main__":
    from pathlib import Path
    import constants
    SYSTEM_PROMPT_FILE_NAME = constants.SYSTEM_PROMPT_4

    base_dir = Path(__file__).parent.parent
    system_prompt_path = base_dir / "data" / "prompts" / SYSTEM_PROMPT_FILE_NAME
    system_prompt = create_system_prompt(system_prompt_path, constants.RectalCancerStagingData)
    print(system_prompt)