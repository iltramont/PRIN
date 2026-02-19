from pydantic import BaseModel
from model_utils import get_regression_fields, get_optional_regression_fields, unwrap_type, field_is_flag_model
import json
from constants import AnnotatedRectalCancerReport




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
    