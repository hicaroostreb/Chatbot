import os

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")


class AIBot:
    def __init__(self):
        self.__chat = ChatGroq(model="llama-3.3-70b-versatile")
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = "/app/chroma_data"
        embedding = HuggingFaceEmbeddings()

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vector_store.as_retriever(
            search_kwargs={"k": 10},
        )

    def __build_messages(self, history_messages, question):
        messages = []
        for message in history_messages:
            # Aqui verificamos se 'message' é um dicionário e se contém o campo "fromMe"
            if isinstance(message, dict) and "fromMe" in message:
                message_class = HumanMessage if message.get("fromMe") else AIMessage
                content = message.get("body")  # Pega o conteúdo da mensagem
            else:
                # Se for uma string, trataremos como mensagem de humano
                message_class = HumanMessage
                content = message  # O conteúdo é a própria string

            # Adiciona a mensagem à lista de mensagens com a classe apropriada
            messages.append(message_class(content=content))

        # Adicionar a pergunta final do humano
        messages.append(HumanMessage(content=question))

        return messages

    def invoke(self, history_messages, question):
        SYSTEM_TEMPLATE = """
        Você é um assistente especializado em tirar dúvidas sobre consórcios.
        Seu objetivo é fornecer respostas curtas, objetivas e baseadas principalmente nas informações do contexto abaixo.

        Dê prioridade ao conteúdo fornecido pela base de conhecimento do consórcio e, se possível, evite respostas extensas ou explicações
        desnecessárias. Use o que está no contexto como base principal e, quando necessário, complemente com informações claras e concisas.

        Se o contexto não fornecer informações diretas, faça o mínimo possível de suposições, e evite criar respostas longas ou imprecisas.

        Responda sempre em português brasileiro.

        <context>
        {context}
        </context>
        """

        docs = self.__retriever.invoke(question)
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        document_chain = create_stuff_documents_chain(
            self.__chat, question_answering_prompt
        )
        response = document_chain.invoke(
            {
                "context": docs,
                "messages": self.__build_messages(history_messages, question),
            }
        )
        return response
