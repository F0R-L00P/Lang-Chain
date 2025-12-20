import transformers
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)

llm.invoke("Explain the theory of relativity in simple terms.")

# -------------------------------------------------------------
# --------------------Prompt templates--------------------------
from langchain_core.prompts import PromptTemplate

templat = """Explain the following concept in simple terms: {concept}"""
prompt_template = PromptTemplate.from_template(template=templat)

prompt = prompt_template.invoke({"concept": "Quantum Mechanics"})
print(prompt)

# lets interate the templat with an llm
# to integrate them use LCEL
# NOTE: LCEL = LangChain Expression Language
# the pipe operator "|"" creates a chain connectign different components

llm_chain = prompt_template | llm

# now we can invoke the chain
concept = "Black Holes"
response = llm_chain.invoke({"concept": concept})
print(f"Topic: {concept}\n\n{response}")

# -------------------------------------------------------------
# ----------------------- CHAT Models -------------------------
# -------------------------------------------------------------
# Chat roles:
# System role: Sets the behavior of the MODEL
# human role: The USER input
# AI role: The MODEL response, such as additional examples the model can learn from
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a simple calculator."),
        ("human", "Answer the following question: what is eight plus eight?"),
        ("ai", "Sure! Here is the answer: 8+8=16"),
        # curly brackets indicate input variables
        ("human", "Answer this math question: {math}"),
    ]
)

# define chain
chat_chain = template | llm
# ask question
math = "What is twelve multiplied by twelve?"
response = chat_chain.invoke({"math": math})
print(response)

# -------------------------------------------------------------
# ------------------ FewShotPromptTemplate --------------------
# -------------------------------------------------------------
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

# Few-shot examples that demonstrate the task pattern
examples = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa"},
]

# How each example is formatted
example_prompt = PromptTemplate.from_template("Question: {question}\nAnswer: {answer}")

# Prefix and suffix wrap the examples and the new user input
prefix = "You are a helpful assistant that answers geography questions.\nHere are some examples:"
suffix = "Now answer the new question.\nQuestion: {question}\nAnswer:"

# Build the FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["question"],
)

# Create a chain with the few-shot prompt
fewshot_chain = few_shot_prompt | llm

# Invoke the chain with a similar-style question
question = "What is the capital of Germany?"
response = fewshot_chain.invoke({"question": question})
print(f"Question: {question}\n\n{response}")

# -------------------------------------------------------------
