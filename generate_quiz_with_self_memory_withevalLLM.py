from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils import print_quiz, initialize_response_schemas, initialize_parser, Memory, write_quiz_to_file, align_answer
import argparse
from loguru import logger

from eval_llm import EvalLLM, eval_template

#### This is adjusted code with Difficulty level from 1-5

template = """You are a KnowledgeStore which generates MCQ questions of desired difficulty on a given topic.
You can generate really good quality questions which are different from the questions generated previously. The difficulty scale for you is 1-5. 1 being extremely easy and 5 being extremely hard.

Previously Generated Questions.
{chat_history}

Human: Generate a single MCQ type question of {difficulty} difficulty level focussed on {subject}.
The question should be totally different from previous generated quesions mentioned above.
The question should present a significant challenge, and should ideally incorporate real-world applications or data to contextualize the mathematical or statistical concepts involved. 
Ensure the question demand a deep understanding and application of theoretical principles, possibly involving multiple steps or the integration of several concepts.
Ensure that Question is strictly multiple choice question with exactly 4 choices which are Choice1, Choice2, Choice3, Choice4. The answer should definitely be one of the Choices.
It is essential to make sure you generate a single question even if you are not able to meet all the requirments.
You can ignore other requirements if not met but it is compulsory to be a single MCQ question with 4 Choices.

{format_instructions}
Output:
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quiz Generation Module')
    parser.add_argument("--role", type=str, default='teacher')
    parser.add_argument("--file-path" , type=str)
    parser.add_argument("--num-retries" , type=int, default=5)

    args = parser.parse_args()
    mode = args.role
    file_path = args.file_path
    num_retries = args.num_retries
    logger.info(f"Mode : {mode}")

    
    # Load the Mistral Model 
    logger.info("Loading Model")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    logger.info("Initializing Output Parser")
    response_schemas = initialize_response_schemas()
    output_parser, format_instructions = initialize_parser(response_schemas)

    # Initialize Memory 
    memory = Memory()

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
            
        with open(file_path,'w') as f:
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

        eval_llm = EvalLLM(template = eval_template, hf_pipeline = hf)

        prompt = PromptTemplate.from_template(template)
        prompt = PromptTemplate(
            input_variables=["chat_history", 
                             "difficulty",
                             "subject",
                             "format_instructions"], template=template
        )    
        # Creating Chain 
        prompt_template = PromptTemplate.from_template(template)
        llm_chain = LLMChain(
                    llm=hf,
                    prompt=prompt,
                    verbose=False
                )

        quiz = []
        

        question_num = 1
        retry = 0
        while question_num <=num_questions:
            try:
                logger.info(f"Question Number: {question_num}")
                logger.info(f"Chat History : {memory.chat_history}")
                input_dict = {
                    "chat_history" : memory.chat_history,
                    "difficulty" : difficulty,
                    "subject" : subject,
                    "format_instructions" : format_instructions
                }
                gen_ques = llm_chain.invoke(input_dict).get('text')
                gen_ques = output_parser.parse(gen_ques)
                gen_ques = align_answer(gen_ques)
                gen_ques = eval_llm.eval_response(data = gen_ques, verbose=True)
                
                question_text = gen_ques.get('Question')
                is_updated = memory.update(chat=question_text)
                if is_updated:
                    question_num+=1
                    quiz.append(gen_ques)
                retry=0
            except Exception as e:
                logger.info(f"Failed to Generate Question Number : {question_num}")
                logger.info(f"Error : {e}")
                logger.info("Retrying")
                retry+=1
                if retry == num_retries:
                    logger.info("Error Occured! Stopping")
                    break
        if mode == 'teacher':
            #quiz = align_answers(quiz)
            write_quiz_to_file(quiz, file_path)
    

    

    