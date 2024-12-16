import os
from tqdm import tqdm
from groq import Groq
import time


class DataCreator:
    def __init__(self):
        self.client = Groq(
            api_key=os.environ["GROQ_RESEARCH"],
        )

        self.system_prompt = "you are a helpful translator"

    def translate_text_data(self, dataframe, target_language, data_dir):
        tokens_list = dataframe["tokens"].tolist()
        sentences_list = [" ".join(tokens) for tokens in tokens_list]

        max_calls_per_minute = 10
        call_timestamps = []
        buffer = 2

        for sentence in tqdm(sentences_list):
            # Remove timestamps older than 1 minute
            current_time = time.time()
            call_timestamps = [ts for ts in call_timestamps if current_time - ts < 60]

            if len(call_timestamps) >= max_calls_per_minute:
                sleep_time = 60 + buffer - (current_time - call_timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            translation = self.translate_sentence(sentence, target_language)
            call_timestamps.append(time.time())

            with open(data_dir, "a", encoding="utf-8") as f:
                f.write("[START]" + translation + "[END]" + "\n")

    def translate_sentence(self, translate_text, target_language):
        prompt = self.format_prompt(translate_text, target_language)
        llm_response = self.prompt_llm(prompt)

        return llm_response

    def prompt_llm(self, prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama3-70b-8192",
        )

        return chat_completion.choices[0].message.content

    def format_prompt(self, translate_text, target_language):
        prompt = f"""
            Provide a high-quality, fluent translation of the following text into {target_language}. 
            Do not interpret, reformat, or list any part of the text. 
            Do not include additional comments or notes.
            Do not output multiple lines of text.
            Do not include explanations, justifications, or meta-commentary about the translation.
            Proper nouns, technical terms, or elements that do not require translation should remain unchanged without further note.
            
            The translation should:

            - Accurately convey the original meaning and intent.
            - Maintain the tone, style, and nuances of the original text.
            - Be natural and culturally appropriate in the target language.

            TEXT: {translate_text}

            Respond only with the translated text and nothing else.
            ONLY INCLUDE THE TRANSLATED TEXT IN {target_language} AS THE RESPONSE
        """

        return prompt
