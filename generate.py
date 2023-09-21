import os
import json
from kendra_index_retriever import KendraIndexRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate


class DocumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Document):
            # Convert the Document object to a dictionary representation
            return obj.__dict__
        return super().default(obj)


def build_chain():
    region = os.environ["AWS_REGION"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=600)

    retriever = KendraIndexRetriever(
        kendraindex=kendra_index_id, awsregion=region, return_source_documents=True
    )

    prompt_template = """
    You work for the company Micropole/Lucy in the cloud as an internal expert. You possess extensive knowledge and expertise 
    about various aspects of the company, including operations, policies, customers, products, services,
    and other relevant information. Your task is to provide guidance and share your knowledge with others 
    within the organization. You are talkative and provides lots of specific details from its context.
    Instruction: Based on context : {context}
    Provide a detailed answer for the question : {question} 
    Answer "I don't know" if not present in the document. Give answers only based on documents but don't quote sources. 
    Solution:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )


def run_chain(chain, prompt: str, history=[]):
    result = chain(prompt)

    # Serialize the source_documents using the custom JSON encoder
    serialized_documents = json.dumps(result["source_documents"], cls=DocumentEncoder)

    # Create a new result dictionary with the serialized source_documents
    serialized_result = {
        "answer": result["result"],
        "source_documents": serialized_documents,
    }

    return serialized_result


def lambda_handler(event, context):
    # Assuming the prompt is passed as a part of the event object
    prompt = event["user_prompt"]

    chain = build_chain()
    result = run_chain(chain, prompt)

    return json.dumps(result)
