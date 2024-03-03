from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser, CommaSeparatedListOutputParser
import json
import re

import sys

template = """
Generate a Quiz with {num_questions} {difficulty} MCQ type questions for graduate students focused on the topic of {subject}. 
Each question should present a significant challenge, aligning with master's level coursework, and should ideally incorporate real-world applications or data to contextualize the mathematical or statistical concepts involved. 
Ensure the questions demand a deep understanding and application of theoretical principles, possibly involving multiple steps or the integration of several concepts.
Ensure that Questions are strictly multiple choice questions and each question should have 4 choices strictly. The answer should be a choice name only. Eg Choice1, or Choice2 etc

The answer should definitely be one of the Choices.

{format_instructions}
"""

def initialize_response_schemas():
    """Initialises response schemas for StructuredOutputParser"""
    question_schema = ResponseSchema(name='Question', description="Question Generated on the given topic")
    choice1_schema = ResponseSchema(name='Choice1', description='Choice 1 for the given question')
    choice2_schema = ResponseSchema(name='Choice2', description='Choice 2 for the given question')
    choice3_schema = ResponseSchema(name='Choice3', description='Choice 3 for the given question')
    choice4_schema = ResponseSchema(name='Choice4', description='Choice 4 for the given question')
    answer_schema = ResponseSchema(name='Answer', description='One of the selected choices out of 4 choices given as the answer')
    explanation_schema = ResponseSchema(name='Explanation', description = 'Explanation why a particular choice is selected as the answer')
    
    response_schemas = [question_schema, 
                        choice1_schema,
                        choice2_schema,
                        choice3_schema,
                        choice4_schema,
                        answer_schema,
                        explanation_schema]
    return response_schemas

def initialize_parser(response_schemas):
    """Initialise output parser and create format instructions for LLM"""
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    return output_parser, format_instructions


def format_output(llm_output, output_parser):
    """Format the LLM Output into List format"""
    output_response = []
    pattern = r"```json\s+(.+?)\s+```"

    # Find all matches
    matches = re.findall(pattern, generated_quiz, re.DOTALL)
    
    # Print matches
    for match in matches:
        output_response.append(output_parser.parse(match))
    return output_response

def print_quiz(quiz):
    for question in quiz:
        print(f"Question:{question.get('Question')}")
        print(f"Choice1:{question.get('Choice1')}")
        print(f"Choice2:{question.get('Choice2')}")
        print(f"Choice3:{question.get('Choice3')}")
        print(f"Choice4:{question.get('Choice4')}")
        print(f"Answer:{question.get('Answer')}")
        print(f"Explanation:{question.get('Explanation')}")
        print()
    

if __name__ == "__main__":

    """
    # How many questions can it generate 
    # Evaluation LLM 
    # Consistency Check : P1
    # Generic Outputs
    # Adjust Difficulty per question according to user's performance
    # Pydentic to return response 
    # https://python.langchain.com/docs/modules/model_io/output_parsers/types/pydantic
    """
    
    mode = sys.argv[1]
    assert mode in ('teacher', 'student')
    
    # Load the Mistral Model 
    print("Loading Model")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    print("Initializing Output Parser")
    response_schemas = initialize_response_schemas()
    output_parser, format_instructions = initialize_parser(response_schemas)
    
    print("Enter -1 to exit")
    while True:
        # User Inputs
        subject = input("Enter the subject:")
        if subject == "-1":
            break
        difficulty = input("Enter the Difficuly of the Questions:")
        if difficulty == "-1":
            break
        num_questions = int(input("Number of Questions:"))
        if num_questions == -1:
            break
            
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

        generated_quiz = chain.invoke({
            "num_questions" : num_questions,
            "difficulty": difficulty,
            "subject": subject,
            "format_instructions" : format_instructions
        })
        formatted_quiz = format_output(llm_output=generated_quiz,
                                       output_parser=output_parser)

        if mode == 'teacher':
            print_quiz(formatted_quiz)
        else:
            print("Lets Play Quizzer")
            
            

        
