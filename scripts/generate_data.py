import os
import json
from openai import OpenAI


class DataGenerator:
    def __init__(self):
        self.api_key = os.environ["OPENAI_RESEARCH"]
        self.client = OpenAI(api_key=self.api_key)

        self.current_batch = None

    def process_batch_request(self, batch_requests_dir: str):
        batch_input_file = self.client.files.create(
            file=open(batch_requests_dir, "rb"), purpose="batch"
        )

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        self.current_batch = batch_object

        print(f"Created batch request. ID: {batch_object.id}")

    def save_batch_response(self, output_dir: str):
        if self.batch_completed():
            file_response = self.client.files.content(self.current_batch.output_file_id)

            jsonl_lines = file_response.text.strip().split("\n")

            with open(output_dir, "a") as outfile:
                for line in jsonl_lines:
                    json_object = json.loads(line)
                    response_message = json_object["response"]["body"]["choices"][0][
                        "message"
                    ]["content"]
                    outfile.write(response_message + "\n")

        else:
            print(
                f"Batch not completed yet. Progress: {self.current_batch.request_counts}"
            )

    def batch_completed(self):
        status = self.current_batch.status
        if status == "completed":
            return True
        return False
