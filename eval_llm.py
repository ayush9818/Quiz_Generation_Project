import warnings
warnings.filterwarnings('ignore')

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser, CommaSeparatedListOutputParser

from loguru import logger
from statistics import mode 

eval_template = """Role: You are an expert grader. You can evaluate the Quizzes' answers. Given MCQ Quiz Question along with the answer,
you need to check the answer and provide the correct choice of answer if it is wrong.

User: Below is the input MCQ question with 4 choices and a user-given answer with Explanation. 
Evalaute the answer and give correct choice if answer is wrong. You should not give your chain of though in the response.
Just give the final correct choice of answer along with a brief explanation. You should only output "Choice1", "Choice2", "Choice3" or "Choice4" in final Answer, don't output any of their contents.

Input: 
Question: {Question} 
Choice1 : {Choice1}
Choice2 : {Choice2}
Choice3 : {Choice3}
Choice4 : {Choice4}
Answer  : {Answer}
Explanation : {Explanation}

{format_instructions}
"""

class EvalLLM:
    def __init__(self, template, hf_pipeline):
        self.template = template
        self.hf_pipeline = hf_pipeline 
        self.response_schemas = self.init_schema()
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.prompt = self.init_prompt(), 
        self.eval_chain = LLMChain(llm=hf_pipeline,
                                    prompt = self.prompt[0], 
                                    output_parser = self.output_parser)

    def init_schema(self):
        """Initialises response schemas for StructuredOutputParser"""
        answer_schema = ResponseSchema(name='Answer', description="Correct Answer to the given Question")
        explanation_schema = ResponseSchema(name='Explanation', 
                                description = 'Explanation why a particular choice is selected as the answer')
        
        response_schemas = [answer_schema,
                            explanation_schema]
        return response_schemas

    def init_prompt(self):
        prompt = PromptTemplate.from_template(self.template)
        return prompt

    def eval_response(self, data, num_evals=3, verbose=False):
        choices_list = [data['Answer']]
        explanation_list = [data['Explanation']]
        data['format_instructions'] = self.format_instructions
        for eval in range(num_evals):
            logger.info(f"Evaluating {eval+1}")
            try:
                eval_response = self.eval_chain.invoke(data).get('text')
                choice = eval_response['Answer']
                explanation = eval_response['Explanation']
    
                choices_list.append(choice)
                explanation_list.append(explanation)
            except:
                logger.error(f"Evaluation {eval+1} Failed")
                continue

        if verbose:
            logger.info(f"Eval Choices : {choices_list}")
            logger.info(f"Eval Explanations : {explanation_list}")
        # Taking Mode of the Response 
        best_answer_choice = mode(choices_list)
        choice_index = choices_list.index(best_answer_choice)
        best_explanation = explanation_list[choice_index]

        data['Answer'] = best_answer_choice
        data['Explanation'] = best_explanation

        return data

        
            

        