import os
from groq import Groq


class DataCreator:
    def __init__(self):

        self.client = Groq(
            api_key=os.environ["GROQ_RESEARCH"],
        )

        self.system_prompt = "you are a helpful translator"

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
            Produce a high-quality, contextually accurate translation that:

            Preserves the original meaning and intent
            Sounds natural in the target language
            Respects cultural and linguistic nuances
            Maintains the original text's tone and style

            TARGET LANGUAGE: {target_language}

            TEXT to TRANSLATE: {translate_text}

            Only respond with the translated text, nothing else.
        """

        return prompt

dc = DataCreator()

sentence = "He has also won the Japanese Senior Open twice."

print(dc.translate_sentence(sentence, "Asturian"))
