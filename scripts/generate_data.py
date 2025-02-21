import os
import ast
from tqdm import tqdm
import pandas as pd
from openai import OpenAI


def convert_to_list(string_representation: str) -> list:
    return ast.literal_eval(string_representation)


class DataGenerator:
    def __init__(self):
        self.api_key = os.environ["OPENAI_RESEARCH"]
        self.client = OpenAI(api_key=self.api_key)

    def generate_data(
        self, template_df_dir: str, languages: dict, num_rows: int, output_dir: str
    ):

        for lang_code, language_name in languages.items():
            output_dir = f"{output_dir}{lang_code}_texts.txt"

            template_df = pd.read_csv(template_df_dir)
            template_df["sentences"] = template_df["tokens"].apply(
                lambda row: " ".join(convert_to_list(row))
            )
            sentence_list = template_df["sentences"].tolist()[:num_rows]

            with open(output_dir, "a", encoding="utf-8") as outfile:
                for sentence in tqdm(sentence_list, desc="Generating data"):
                    translation = self.translate_sentence(sentence, language_name)
                    outfile.write(f"{translation}\n")

    def translate_sentence(self, sentence: str, target_language: str):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence, and your task is to translate it into {target_language}. "
                    "Provide only the translation with no explanations, notes, or disclaimers. "
                    "If a word is a proper noun or untranslatable, transliterate it or leave it unchanged.",
                },
                {"role": "user", "content": sentence},
            ],
        )

        return completion.choices[0].message.content

data_generator = DataGenerator()

languages = {
    "fo": "Faroese",
    "co": "Corsican",
    "hsb": "Upper Sorbian",
    "bh": "Bhojpuri",
    "cv": "Chuvash",
    "mg": "Malagasy"
}

data_generator.generate_data(
    "data/labaled/en_data.csv", languages, 10000, "data/unlabeled/"
)