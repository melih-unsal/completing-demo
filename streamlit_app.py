import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Add API key input to sidebar
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
st.sidebar.markdown("""
If you don't have an OpenAI API key, you can get one by:
1. Going to [OpenAI's website](https://platform.openai.com/account/api-keys)
2. Creating an account or signing in
3. Clicking on "Create new secret key"
""")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Initialize embedding model with the API key from sidebar
embedding_model = OpenAIEmbeddings(
    openai_api_key=api_key,
    model="text-embedding-3-large"
)

vectorstore = PineconeVectorStore(
    index_name="completing",
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 8, "score_threshold": 0.5})

llm = ChatOpenAI(model="o4-mini", openai_api_key=api_key)


def answer(question):
    
    chat_history_prompt_template = """
    You are a helpful assistant that can create a question based on the chat history and the user question.
    Don't extrapolate the question, just create a question based on the chat history and the user question.
    Make it concise and to the point.
    Chat History: {chat_history}
    User Question: {question}
    Question:
    """
    
    chat_history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            chat_history_prompt_template,
        )
    ]
    )
    
    chain = chat_history_prompt | llm

    response = chain.invoke({"question": question, "chat_history": st.session_state.messages})
    
    optimized_question = response.content
    
    st.write(optimized_question)
    
    
    retrieved_documents = retriever.invoke(optimized_question)
    context = "\n".join([doc.page_content for doc in retrieved_documents])

    prompt_template = """
    You are a helpful assistant that can positioned as an AI buddy that can help the team members about the Company called "Completing".
    You are given a question and a context.
    If you need to give the completing's email adress use done@completing.com
    Those are the clients names:
    - Diabetes Questions and Answers (DQ&A)
    - Abstrax Tech
    - AMC Modern IT
    - Talonvest Capital Inc. 
    - iTrust Capital 
    - The Students Commission of Canada (SCC)
    - Ninth Platform
    - Do Yoga With Me
    - The FeedFeed
    - Virtual Summits
    - SPH Medical 
    - NonStop Reviews
    
    You should give the requested information about the company "Completing" and by this way, each team member can learn more about the company.
    Question: {question}
    Context: {context}
    """
    
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            prompt_template,
        )
    ]
)


    chain = prompt | llm

    response = chain.invoke({"question": question, "context": context})

    return response.content

st.title("Ask about Completing")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know about Completing?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.spinner("Thinking..."):
        response = answer(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
