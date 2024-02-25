from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import re


template = """
Generate {num_questions} {difficulty} quiz questions for graduate students focused on the topic of {subject}. 
Each question should present a significant challenge, aligning with master's level coursework, and should ideally incorporate real-world applications or data to contextualize the mathematical or statistical concepts involved. 
Ensure the questions demand a deep understanding and application of theoretical principles, possibly involving multiple steps or the integration of several concepts.

The answer should definitely be one of the Choices.

Here is the desired JSON structure for each question:

Output Format:

Question: [Insert Question Here]
Choice1: [Insert Choice1 Here]
Choice2: [Insert Choice2 Here]
Choice3: [Insert Choice3 Here]
Choice4: [Insert Choice4 Here]
Answer: [Correct choice out of the 4 given choices]
Explanation : [Explanation of the correct choice]
"""


class QuizQuestionJSONFormatter:
    def __init__(self, text_input):
        self.text_input = text_input

    def parse_questions(self):
        # Split the text input into chunks based on the question pattern
        question_blocks = re.split(r'Question \d+:', self.text_input)[1:]
        return question_blocks

    def extract_choices(self, block):
        # Extract choices from the block
        choices_pattern = r'Choice\d+: ([^\n]+)'
        choices = re.findall(choices_pattern, block)
        return choices

    def format_json(self):
        # Parse each question block and format it into JSON
        question_blocks = self.parse_questions()
        formatted_questions = []

        for block in question_blocks:
            # Extract question, answer, and explanation using regular expressions
            question = re.search(r'([^\n]+)', block).group(1).strip()
            choices = self.extract_choices(block)
            answer = re.search(r'Answer: (Choice\d+)', block).group(1).strip()
            explanation = re.search(r'Explanation: ([^\n]+)', block).group(1).strip()

            # Build the JSON object
            question_json = {
                "Question": question,
                "Choices": choices,
                "Answer": answer,
                "Explanation": explanation
            }
            formatted_questions.append(question_json)

        return json.dumps(formatted_questions, indent=4)

if __name__ == "__main__":

    # Load the Mistral Model 
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    # User Inputs
    subject = input("Enter the subject:")
    difficulty = input("Enter the Difficuly of the Questions:")
    num_questions = int(input("Number of Questions:"))

    # Defining HuggingFace Pipeline
    # TODO : Put generation parameters in config
    pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer,
                device=0, 
                max_new_tokens=2000,
                do_sample=True, 
                top_k=20, 
                top_p=0.7,
                early_stopping=True,
                num_beams=2
               )
    hf = HuggingFacePipeline(pipeline=pipe)

    # Creating Chain 
    prompt_template = PromptTemplate.from_template(template)
    chain = prompt_template | hf

    print("Generating Quiz...................................")
    generated_quiz = chain.invoke({
        "num_questions" : num_questions,
        "difficulty": difficulty,
        "subject": subject
    })
    print(generated_quiz)
    print("Formatting Quiz")

    # TODO : Look for more Robust Formatter
    question_formatter = QuizQuestionJSONFormatter(text_input=generated_quiz)
    formatted_quiz = json.loads(question_formatter.format_json())
    print(formatted_quiz)


    