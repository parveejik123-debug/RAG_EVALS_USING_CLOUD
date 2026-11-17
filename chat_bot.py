import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

from langsmith import Client

client = Client()



# Define dataset
dataset_name = "Chatbots Evaluation"

# Check whether dataset exists
existing_dataset = client.has_dataset(dataset_name=dataset_name)

if existing_dataset:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print("Dataset already exists. Reusing existing dataset.")

else:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Dataset for chatbot evaluation"
    )
    print("Dataset created successfully.")


# Define dataset: these are your test cases
# dataset_name = "Chatbots Evaluation"
# dataset = client.create_dataset(dataset_name)
# client.create_examples(
#     dataset_id=dataset.id,
#     examples=[
#         {
#             "inputs": {"question": "What is LangChain?"},
#             "outputs": {"answer": "A framework for building LLM applications"},
#         },
#         {
#             "inputs": {"question": "What is LangSmith?"},
#             "outputs": {"answer": "A platform for observing and evaluating LLM applications"},
#         },
#         {
#             "inputs": {"question": "What is OpenAI?"},
#             "outputs": {"answer": "A company that creates Large Language Models"},
#         },
#         {
#             "inputs": {"question": "What is Google?"},
#             "outputs": {"answer": "A technology company known for search"},
#         },
#         {
#             "inputs": {"question": "What is Mistral?"},
#             "outputs": {"answer": "A company that creates Large Language Models"},
#         }
#     ]
# )

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama_client = OpenAI(
    base_url = OLLAMA_BASE_URL,
    api_key = "llama3.2"
)

from langsmith import wrappers
 
openai_client=wrappers.wrap_openai(ollama_client)

eval_instructions = "You are an expert professor specialized in grading students' answers to questions."

def correctness(inputs:dict,outputs:dict, reference_outputs:dict)->bool:
      user_content = f"""You are grading the following question:
    {inputs['question']}
    Here is the real answer:
    {reference_outputs['answer']}
    You are grading the following predicted answer:
    {outputs['response']}
    Respond with CORRECT or INCORRECT:
    Grade:
    """
      response=openai_client.chat.completions.create(
            model="llama3.2",
            temperature=0,
            messages=[
                  {"role":"system","content":eval_instructions},
                  {"role":"user","content":user_content}
            ]
      ).choices[0].message.content

      print("QUESTION:", inputs["question"])
      print("REFERENCE:", reference_outputs["answer"])
      print("PREDICTED:", outputs["response"])
      print("JUDGE RESPONSE:", repr(response))


      return response.strip().upper() == "CORRECT"

## Concisions- checks whether the actual output is less than 2x the length of the expected result.

def concision(outputs: dict, reference_outputs: dict) -> bool:
    return int(len(outputs["response"]) < 2 * len(reference_outputs["answer"]))

default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."
def my_app(question: str, model: str = "llama3.2", instructions: str = default_instructions) -> str:
    return openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content

### Call my_app for every datapoints
def ls_target(inputs: str) -> dict:
    return {"response": my_app(inputs["question"])}

## Run our evaluation
experiment_results=client.evaluate(
    ls_target, ## Your AI system
    data=dataset_name,
    evaluators=[correctness,concision],
    experiment_prefix="ollama3.2-chatbot"
)