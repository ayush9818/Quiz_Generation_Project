from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory, ConversationBufferWindowMemory

import argparse
from utils import initialize_response_schemas, initialize_parser, print_quiz, write_quiz_to_file
import json
import re
from loguru import logger
import sys

template = """You are a KnowledgeStore which generates MCQ questions of desired difficulty on a given topic.
You can generate really good quality questions which are different from the questions generated previously.

Previously Generated Questions.
{chat_history}

Human: Generate a single MCQ type question of {difficulty} difficulty level focussed on {subject}.
The question should be totally different from previous generated quesions mentioned above.
The question should present a significant challenge, and should ideally incorporate real-world applications or data to contextualize the mathematical or statistical concepts involved. 
Ensure the question demand a deep understanding and application of theoretical principles, possibly involving multiple steps or the integration of several concepts.
Ensure that Question is strictly multiple choice question with exactly 4 choices. 
Choice1, Choice2, Choice3, Choice4. The answer should definitely be one of the Choices.
It is essential to make sure you generate a single question even if you are not able to meet all the requirments.
You can ignore other requirements if not met but it is compulsory to be a single MCQ question with 4 Choices.

{format_instructions}
Output:
"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='teacher')
    parser.add_argument("--memory-buffer" , type=int, default=40)
    parser.add_argument("--file-path" , type=str)
    args = parser.parse_args()

    mode = args.mode
    memory_buffer = args.memory_buffer
    file_path = args.file_path
    logger.info(f"Mode : {mode}, Memory Buffer : {memory_buffer}")
    assert mode in ('teacher', 'student')
    
    # Load the Mistral Model 
    logger.info("Loading Model")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    logger.info("Initializing Output Parser")
    response_schemas = initialize_response_schemas()
    output_parser, format_instructions = initialize_parser(response_schemas)
    
    logger.info("Enter -1 to exit")
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

        with open(file_path,'a') as f:
            f.write(f"Subject : {subject}\nDifficulty: {difficulty}\n\n")
        
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
        memory = ConversationBufferMemory(k=memory_buffer, memory_key="chat_history", input_key='subject')
    
        # Creating Chain 
        prompt_template = PromptTemplate.from_template(template)
        llm_chain = LLMChain(
                    llm=hf,
                    prompt=prompt,
                    memory=memory,
                    verbose=False
                )
        chat_history_list = []
        chat_history_str = ''

        
        quiz = []
        question_num = 1
        retry = 0
        while question_num <=num_questions:
            try:
                input_dict = {
                        "difficulty" : difficulty,
                        "subject" : subject,
                        "format_instructions" : format_instructions,
                        "chat_history" : chat_history_str
                }
                logger.info(f"Generating Question Number : {question_num}")
                logger.info(f"Current History : {chat_history_str}")
                gen_ques = llm_chain.invoke(input_dict).get('text')
                gen_ques = output_parser.parse(gen_ques)

                question = gen_ques["Question"]
                if question not in chat_history_list:
                    chat_history_list.append(question)
                    chat_history_str+=f"{question}\n"
                    question_num+=1
                    quiz.append(gen_ques)
                retry=0
            except Exception as e:
                logger.info(f"Error: {e}")
                logger.info(f"Failed to Generate Question Number : {question_num}. Retrying.......")
                retry+=1
                if retry == 3:
                    logger.info("Error Occured! Stopping")
                    break
        if mode == 'teacher':
            #print_quiz(quiz)
            write_quiz_to_file(quiz, file_path)
            
            

        
