from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import re

from Llm.llm_loader import LLM

project_root = Path.cwd().resolve()

TIMESTAMP_PATTERN = re.compile(r"iterative_database_retrival_MMQA_(\d{8}_\d{6})\.json$")


def append_log_entry(log_records, row, response_text,Answer_llm,save_path):
    log_records.append(
        {
            'model': Answer_llm.model_name,
            'provider': Answer_llm.provider,
            'id': f"{row['id']}",
            'question': row['question'],
            'spider_db_id': row['spider_db_id'],
            'predict_db_id': row['predict_db_id'],
            'predict_schema_linking': response_text,
        }
    )
    save_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding='utf-8')


def extract_timestamp(path: Path) -> str | None:
    match = TIMESTAMP_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1)


def find_model_dir(logs_dir: Path, model_name: str) -> Path:
    direct_path = logs_dir / Path(model_name)
    if direct_path.is_dir():
        return direct_path

    matching_dirs = sorted(
        path for path in logs_dir.rglob(model_name) if path.is_dir() and path.name == model_name
    )

    if not matching_dirs and "/" in model_name:
        leaf_name = Path(model_name).name
        matching_dirs = sorted(
            path for path in logs_dir.rglob(leaf_name) if path.is_dir() and path.name == leaf_name
        )

    if not matching_dirs:
        raise FileNotFoundError(
            f"Could not find a model directory named '{model_name}' under {logs_dir}."
        )
    if len(matching_dirs) > 1:
        matched_paths = "\n".join(str(path) for path in matching_dirs)
        raise ValueError(
            f"Found multiple model directories named '{model_name}'. Please disambiguate:\n{matched_paths}"
        )
    return matching_dirs[0]

def find_result_file(model_dir: Path, timestamp=None) -> Path:
    candidate_files = []
    for path in model_dir.glob("iterative_database_retrival_MMQA_*.json"):
        file_timestamp = extract_timestamp(path)
        if file_timestamp is None:
            continue
        candidate_files.append((file_timestamp, path))

    if not candidate_files:
        raise FileNotFoundError(
            f"Could not find iterative_database_retrival_MMQA_*.json under {model_dir}."
        )

    if timestamp is not None:
        for file_timestamp, path in candidate_files:
            if file_timestamp == timestamp:
                return path
        raise FileNotFoundError(
            f"Could not find iterative_database_retrival_MMQA_{timestamp}.json under {model_dir}."
        )

    candidate_files.sort(key=lambda item: item[0], reverse=True)
    return candidate_files[0][1]

def run_(project_root,prompt_template, save_path, pre_db_path,dataset_name,Answer_llm):
    log_records = []
    save_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2), encoding='utf-8')

    df = pd.read_json(pre_db_path)

    for _,row in tqdm(df.iterrows(),total=len(df)):
        schema_path = f"{project_root}/Data/{dataset_name}/Table_schema_csv/{row['predict_db_id']}.csv"
        schema_df = pd.read_csv(schema_path)
        database_schema = schema_df.to_markdown(index=False)
        prompt = (
            prompt_template
            .replace('{DATABASE_SCHEMA}', database_schema)
            .replace('{QUESTION}', row['question'])
            .replace('{HINT}', 'No hint')
        )
        res = Answer_llm.query(prompt).replace("```","").replace("json","")        
        if "</think>" in res:
            res = res.split("</think>")[-1]      
        append_log_entry(log_records, row, res,Answer_llm, save_path)

    print(f'All responses have been saved to {save_path}')
    

def main() -> None:
    dataset_name = 'MMQA'
    method_ = 'few_shot'

    prompt_path = project_root / 'Templates' / method_ / 'baseline_schema_linking.txt'
    logs_dir = project_root / "Logs"

    tranformers_llm_lists = ['mistralai/Ministral-3-8B-Instruct-2512','Qwen/Qwen3.5-9B']
    prompt_template = prompt_path.read_text(encoding='utf-8').strip()
    for model_name in tranformers_llm_lists:
        model_dir = find_model_dir(logs_dir=logs_dir, model_name=model_name)
        pre_db_path = find_result_file(model_dir=model_dir)

        Answer_llm = LLM(model_name = model_name, provider= 'transformers')

        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = project_root / 'Logs' / model_name
        save_dir.mkdir(parents=True,exist_ok=True)
        save_path = save_dir / f'{method_}_baseline_schema_linking_{dataset_name}_{run_id}.json'

        run_(project_root,prompt_template, save_path, pre_db_path,dataset_name,Answer_llm)



if __name__ == "__main__":
    main()
