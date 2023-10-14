from llmstudio.models import OpenAIClient
from llmstudio.models.models import LLMCompare


if __name__ == '__main__':

    open_ai_client = OpenAIClient(api_key='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')
    gpt4 = open_ai_client.get_model(model_name='gpt-4')
    gpt3 = open_ai_client.get_model(model_name='gpt-3.5-turbo')

    llm_compare = LLMCompare()
    output_dict = llm_compare.compare(models=[gpt4, gpt3], prompt='Hello, my name is Claudio. I am a Data Scientist.', )
    print(output_dict)


