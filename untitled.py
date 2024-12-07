import streamlit as st
import tiktoken
from loguru import logger

# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

# from langchain.memory import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatOllama

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough 

import os

def main():
    global retriever
    
    st.set_page_config(
        page_title="streamlit_Rag",
        page_icon=":books:")
    st.title("_RAG_test4 :red[Q/A Chat]_ :books:")

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    if 'store' not in st.session_state:
        st.session_state['store'] = dict()

    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = None

    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        process = st.button('Process')

    if process:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.retriever = vectorstore.as_retriever(search_type = 'mmr', verbose=True)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
            st.session_state['messages'] = [{'role': 'assistant',
                                             'content': '안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!'}]

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    RAG_PROMPT_TEMPLATE = '''당신은 동서울대학교 컴퓨터소프트웨어과를 소개하는 전문 상담원입니다.
    아래의 Context와 규칙을 기반으로 질문에 정확하고 상세하게 답변해주세요.
    
    규칙:
    1. Context에 있는 정보만을 사용하여 답변하세요.
    2. Context에 없는 내용은 추측하지 말고 "주어진 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.
    3. 답변은 친절하고 전문적인 톤으로 작성하세요.
    4. 학과 정보에 대해 질문할 경우, 가능한 한 구체적인 정보를 제공하세요.

    Question: {question}
    Context: {context}
    Answer: '''

    print_history()

    if user_input := st.chat_input('메세지를 입력해 주세요'):
        add_history('user', user_input)
        st.chat_message('user').write(f'{user_input}')
        with st.chat_message('assistant'):
            llm = ChatOllama(model = 'llama3.1')
            chat_container = st.empty()
    
            if st.session_state.processComplete == True and st.session_state.retriever is not None:
                prompt1 = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                retriever = st.session_state.retriever  # session state에서 retriever 가져오기
                rag_chain = (
                    {
                        'context': lambda x: format_docs(retriever.get_relevant_documents(x)),
                        'question': RunnablePassthrough()
                    }
                    | prompt1
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.stream(user_input)
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown(''.join(chunks))
                add_history('ai', ''.join(chunks))

            else:
                prompt2 = ChatPromptTemplate.from_template(
                    '다음의 질문에 간결하게 답변해 주세요:\n{input}'
                )
                chain = prompt2 | llm | StrOutputParser()
                answer = chain.stream(user_input)
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown(''.join(chunks))
                add_history('ai', ''.join(chunks))

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name
        with open(file_name, 'wb') as file:
            file.write(doc.getvalue())
            logger.info(f'Uploaded {file_name}')
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=tiktoken_len
    )
    chunks=text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-multitask',
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore):
    llm = ChatOllama(model = 'Llama3_ko_8b_q5', temperature=0)
    conversation_chain=ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
    get_chat_history=lambda h: h,
    return_source_documents=True,
    verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
