from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate

class ReaderLLM:
    def __init__(self):
        PROMPT_TEMPLATE = """
                            Answer the question based only on the following context:

                            {context}

                            ---

                            Answer the question based on the above context: {question}
                            """
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.llmModel = Ollama(model="llama3.1")

    def query_rag(self, ragModel, query_text: str ):
        results = ragModel.retrive(query_text)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        prompt = self.prompt_template.format(context=context_text, question=query_text)
        response_text = self.llmModel.invoke(prompt)
        sources = {doc.metadata.get("id", None) for doc in results}
        formatted_response = f"\nResponse:\n{response_text}\n\nSources: {sources}\n---"
        return formatted_response;
        
