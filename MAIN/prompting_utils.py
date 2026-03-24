from pydantic import BaseModel
from model_utils import get_regression_fields, get_optional_regression_fields, get_multiple_choice_fields
import json
from constants import AnnotatedRectalCancerReport
from typing import get_origin, get_args, Union
from enum import Enum





def unwrap_optional(t):
    """Ritorna (tipo_base, is_optional)."""
    origin = get_origin(t)
    if origin is Union:
        args = get_args(t)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)][0]
            return non_none, True
    return t, False


def generate_prompt_schema(model: type[BaseModel]) -> dict:
    reg_fields = get_regression_fields(model)
    optional_reg_fields = get_optional_regression_fields(model)

    result = {}

    for name, field_info in model.model_fields.items():
        field_type, is_optional = unwrap_optional(field_info.annotation)
        # Caso regressione
        if name in reg_fields:
            t = "int"
            if name in optional_reg_fields or is_optional:
                t = f"{t} | null"
            result[name] = t
            continue
        # Caso modello annidato
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            result[name] = generate_prompt_schema(field_type)
            continue
        # Caso Enum
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            values = [e.value for e in field_type]
            result[name] = " | ".join(values)
            continue
        # Caso generico (stringhe, int, ecc.)
        base = field_type.__name__
        if is_optional:
            base = f"{base} | null"
        result[name] = base

    return result


def create_system_prompt(prompt_path: str, annotation_model: type[BaseModel]) -> str:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
        
    schema = json.dumps(generate_prompt_schema(annotation_model), indent=2)
    system_prompt = system_prompt.replace("{schema_json}", schema)
    return system_prompt


def add_examples_to_user_prompt(user_report_text: str, examples: list[AnnotatedRectalCancerReport]) -> str:
    parts = []
    if examples:
        parts.append("## Esempi\n")
        for i, ex in enumerate(examples, 1):
            parts.append(f'### Esempio {i}')
            parts.append(f"**Referto:**\n{ex.report_text}")
            parts.append(f"**Output:**\n{ex.report_data.model_dump_json()}")
        
    parts.append("## Referto da analizzare")
    parts.append(f"**Referto:**\n{user_report_text}")
    parts.append("**Output:**")
    
    return "\n\n".join(parts)

def add_examples_to_system_prompt(system_prompt: str, examples: list[AnnotatedRectalCancerReport]) -> str:
    if examples:
        s = f"\n<esempi>"
        for i, ex in enumerate(examples, 1):
            s = s + f"\n\n<esempio>\n<referto>\n{ex.report_text}\n</referto>"
            s = s + f"\n<output>\n{ex.report_data.model_dump_json(indent=2)}\n</output>\n</esempio>"
        s = s + f"\n\n</esempi>"
        return f'{system_prompt}\n{s}'
    else: 
        return system_prompt
    
    



if __name__ == "__main__":
    from pathlib import Path
    import constants
    import pandas as pd
    import model_utils
    base_dir = Path(__file__).parent.parent

    # Carichiamo i nostri file csv
    file_names = {
        'validation': constants.VALIDATION_SPLIT_FILE_NAME,
        'test': constants.TEST_SPLIT_FILE_NAME,
    }

    paths = {split: base_dir / "data" / file_name for split, file_name in file_names.items()}

    data = dict()
    for split, path in paths.items():
        data[split] = pd.read_csv(path)

    validation_data, test_data = data['validation'], data['test']

    ################################
    # Convert float columns to Int64
    ################################
    float_cols = test_data.select_dtypes("float").columns
    for col in float_cols:
        test_data[col] = test_data[col].round().astype("Int64")
        validation_data[col] = validation_data[col].round().astype("Int64")
        
    # Check duplicatest
    assert set(test_data.id) & set(validation_data.id) == set(), "There are overlapping IDs between test and validation sets!"

    print(f"{len(test_data) = }")
    print(f"{len(validation_data) = }")
    
    user_text = test_data.iloc[0].report_text
    report_data = model_utils.from_series_to_basemodel(test_data.iloc[1], constants.RectalCancerStagingData)
    text = test_data.iloc[1].report_text
    ann_report = AnnotatedRectalCancerReport(report_text=text, report_data=report_data)
    report_data = model_utils.from_series_to_basemodel(test_data.iloc[2], constants.RectalCancerStagingData)
    text = test_data.iloc[2].report_text
    ann_report_2 = AnnotatedRectalCancerReport(report_text=text, report_data=report_data)
    print(add_examples_to_user_prompt(text, [ann_report, ann_report_2]))
    