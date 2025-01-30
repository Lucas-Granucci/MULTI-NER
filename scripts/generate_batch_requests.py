import ast
import json
import pandas as pd


def create_batch_requests(
    language_json_dir: str, data_dir: str, num_rows: int, start_idx: int
) -> None:
    """
    Create batch request jsonl files for all languages
    """
    with open(language_json_dir, "r") as file:
        language_groups = json.load(file)

    for lang_group in language_groups:
        low_resource_code = lang_group["low_resource_code"]
        low_resource_language = lang_group["low_resource_name"]

        create_batch_request(
            low_resource_language, low_resource_code, data_dir, num_rows, start_idx
        )

    print("Batch request generation complete")


def create_batch_request(
    language_name: str, language_code: str, data_dir: str, num_rows: int, start_idx: int
):
    """
    Create batch request jsonl file for OpenAI dataset generation
    """
    template_dataset = pd.read_csv("data/labeled/en_data.csv")
    template_dataset["sentences"] = template_dataset["tokens"].apply(
        lambda row: " ".join(convert_to_list(row))
    )
    sentence_list = template_dataset["sentences"].tolist()[
        start_idx : start_idx + num_rows
    ]

    with open(f"{data_dir}/{language_code}_requests.jsonl", "w") as outfile:
        for idx, sentence in enumerate(sentence_list):
            request_entry = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You will be provided with a sentence, and your task is to translate it into {language_name}. "
                                "Provide only the translation with no explanations, notes, or disclaimers. "
                                "If a word is a proper noun or untranslatable, transliterate it or leave it unchanged."
                            ),
                        },
                        {"role": "user", "content": sentence},
                    ],
                    "max_tokens": 128,
                },
            }

            json.dump(request_entry, outfile)
            outfile.write("\n")


def convert_to_list(string_representation: str) -> list:
    return ast.literal_eval(string_representation)


create_batch_requests(
    language_json_dir="data/languages.json",
    data_dir="data/requests",
    num_rows=2000,
    start_idx=0,
)
