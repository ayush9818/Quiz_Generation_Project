from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser, CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory, ConversationBufferWindowMemory
import json
import re

import sys

template = """You are a KnowledgeStore which generates MCQ questions of desired difficulty on a given topic.
You can generate really good quality questions which are different from the questions generated previously.

{chat_history}

Human: Generate a single MCQ type question for graduate students of {difficulty} difficulty level focussed on {subject}. 
The question should present a significant challenge, aligning with master's level coursework, and should ideally incorporate real-world applications or data to contextualize the mathematical or statistical concepts involved. 
Ensure the question demand a deep understanding and application of theoretical principles, possibly involving multiple steps or the integration of several concepts.
Ensure that Question is strictly multiple choice question with exactly 4 choices.
Choice1, Choice2, Choice3, Choice4
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
    answer_schema = ResponseSchema(name='Answer', description='One of the selected choices out of 4 choices given as the answer. Eg Choice1')
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
    MEMORY_BUFFER=5
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
                    device=5, 
                    max_new_tokens=2000,
                    do_sample=True, 
                    top_k=20, 
                    top_p=0.7,
                    early_stopping=True,
                    num_beams=2
                   )
        hf = HuggingFacePipeline(pipeline=pipe)

        prompt = PromptTemplate.from_template(template)
        prompt = PromptTemplate(
            input_variables=["chat_history", 
                             "difficulty",
                             "subject",
                             "format_instructions"], template=template
        )
        #memory = ConversationSummaryMemory(llm=hf, input_key='subject')
        #memory = ConversationBufferWindowMemory(k=MEMORY_BUFFER, memory_key="chat_history", input_key='subject')
        memory = ConversationBufferMemory(k=MEMORY_BUFFER, memory_key="chat_history", input_key='subject')
    
        # Creating Chain 
        prompt_template = PromptTemplate.from_template(template)
        llm_chain = LLMChain(
                    llm=hf,
                    prompt=prompt,
                    memory=memory,
                    verbose=False
                )

        quiz = []
        input_dict = {
            "difficulty" : difficulty,
            "subject" : subject,
            "format_instructions" : format_instructions
        }

        question_num = 1
        retry = 0
        while question_num <=num_questions:
            try:
                gen_ques = llm_chain.run(input_dict)
                print(gen_ques)
                gen_ques = output_parser.parse(gen_ques)
                print(f"Question Number : {question_num} Done")
                question_num+=1
                quiz.append(gen_ques)
                retry=0
            except Exception as e:
                print(f"Failed to Generate Question Number : {question_num}")
                print("Retrying")
                retry+=1

                if retry == 3:
                    print("Error Occured! Stopping")
                    break
        if mode == 'teacher':
            print_quiz(quiz)
            
            

        
