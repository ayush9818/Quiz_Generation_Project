from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser, CommaSeparatedListOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import json
import re
import sys
from loguru import logger

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

def initialize_response_schemas():
    """Initialises response schemas for StructuredOutputParser"""
    question_schema = ResponseSchema(name='Question', description="Question Generated on the given topic")
    choice1_schema = ResponseSchema(name='Choice1', description='Choice 1 for the given question')
    choice2_schema = ResponseSchema(name='Choice2', description='Choice 2 for the given question')
    choice3_schema = ResponseSchema(name='Choice3', description='Choice 3 for the given question')
    choice4_schema = ResponseSchema(name='Choice4', description='Choice 4 for the given question')
    #answer_schema = ResponseSchema(name='Answer', description='One of the selected choices out of 4 choices given as the answer. Eg Choice1')
    answer_schema = ResponseSchema(name='Answer', description='Correct Answer to the Question: Answer Format => Choice 1 or Choice 2 or Choice 3 or Choice 4')
    explanation_schema = ResponseSchema(name='Explanation', description = 'Explanation why a particular choice is selected as the answer')

    
    response_schemas = [question_schema, 
                        choice1_schema,
                        choice2_schema,
                        choice3_schema,
                        choice4_schema,
                        answer_schema,
                        explanation_schema]
    return response_schemas



def retrieve_choice(choice_list, query):
    model_kwargs = {'device': 'cpu'}
    db = Chroma.from_texts(choice_list, HuggingFaceBgeEmbeddings(model_kwargs=model_kwargs))
    return db.similarity_search(query)[0].page_content.split(':')[0]


def align_answer(question):
    choice_list = [f"{key}:{value}" for key,value in question.items() if 'choice' in key.lower()]
    query = f"Answer:{question['Answer']}"
    question['Original Answer'] = question['Answer']
    question['Answer'] = retrieve_choice(choice_list, query)
    return question
    
def initialize_parser(response_schemas):
    """Initialise output parser and create format instructions for LLM"""
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    return output_parser, format_instructions


def write_quiz_to_file(quiz, file_path):
    logger.info(f"Writing Quiz to {file_path}")
    with open(file_path,'a') as f:
        for num, question in enumerate(quiz):
            f.write(f"Question {num+1}:{question.get('Question')}\n")
            f.write(f"Choice1:{question.get('Choice1')}\n")
            f.write(f"Choice2:{question.get('Choice2')}\n")
            f.write(f"Choice3:{question.get('Choice3')}\n")
            f.write(f"Choice4:{question.get('Choice4')}\n")
            f.write(f"Answer:{question.get('Answer')}\n")
            f.write(f"Original Answer:{question.get('Original Answer')}\n")
            f.write(f"Explanation:{question.get('Explanation')}")
            f.write("\n\n")
            


class Memory:
    def __init__(self):
        self.chat_history = ""
        self.chat_list = []

    def update(self, chat):
        if chat not in self.chat_list:
            self.chat_list.append(chat)
            self.chat_history+=f"{chat}\n"
            return True
        else:
            print("Question already there! Not Updating")
            return False

    

    