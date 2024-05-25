from utils import SEED
import openai
import pandas as pd
import random
import os


class GPT4OProcessor:
    def __init__(self, api_key, user_prompt, dataset_path, abstract_column='abstract',
                 target_column='target', seed=None):
        """
        Initialize the GPT4OProcessor.

        Parameters:
        - api_key (str): OpenAI API key.
        - user_prompt (str): The prompt to concatenate with abstracts.
        - dataset_path (str): Path to the dataset containing abstracts.
        - abstract_column (str): Name of the column containing abstracts in the dataset.
        - target_column (str): Name of the column containing targets associated with abstracts.
        - seed (int): Random seed for reproducibility.
        """
        openai.api_key = api_key
        self.user_prompt = user_prompt
        self.dataset = pd.read_csv(dataset_path)
        self.abstract_column = abstract_column
        self.target_column = target_column
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def get_combined_text(self, abstract):
        """
        Concatenate the user prompt with an abstract.

        Parameters:
        - abstract (str): An abstract from the dataset.

        Returns:
        - str: The combined text.
        """
        return f"{self.user_prompt}\n\n{abstract}"

    def get_few_shot_examples(self):
        """
        Randomly select three examples from the dataset for few-shot learning.

        Returns:
        - list: A list of few-shot examples in the format required by the API.
        """
        examples = random.sample(self.dataset.to_dict('records'), 3)
        few_shot_examples = []
        for example in examples:
            example_text = self.get_combined_text(example[self.abstract_column])
            few_shot_examples.append({
                "role": "user",
                "content": example_text
            })
            few_shot_examples.append({
                "role": "assistant",
                "content": example[self.target_column]
            })
        return few_shot_examples

    def process_abstract(self, abstract):
        """
        Send the combined text to GPT-4 and get the response.

        Parameters:
        - abstract (str): An abstract from the dataset.

        Returns:
        - str: The response from GPT-4.
        """
        combined_text = self.get_combined_text(abstract)
        few_shot_examples = self.get_few_shot_examples()
        messages = few_shot_examples + [
            {"role": "user", "content": combined_text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()

    def process_all_abstracts(self):
        """
        Process all abstracts in the dataset and store the results.

        Returns:
        - pd.DataFrame: The original dataset with an added column for GPT-4 responses.
        """
        responses = []
        for abstract in self.dataset[self.abstract_column]:
            response = self.process_abstract(abstract)
            responses.append(response)

        self.dataset['gpt4_response'] = responses
        return self.dataset

    def save_results(self, output_path):
        """
        Save the processed dataset to a CSV file.

        Parameters:
        - output_path (str): Path to save the output CSV file.
        """
        processed_dataset = self.process_all_abstracts()
        processed_dataset.to_csv(output_path, index=False)


# Example usage
if __name__ == "__main__":
    api_key = "your_openai_api_key_here"
    user_prompt = "PROMPT"
    dataset_path = "path_to_your_dataset.csv"
    output_path = "path_to_save_processed_dataset.csv"

    processor = GPT4OProcessor(api_key, user_prompt, dataset_path, seed=seed)
    processor.save_results(output_path)
