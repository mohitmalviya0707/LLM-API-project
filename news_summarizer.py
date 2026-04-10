from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate  

# Tool
search_tool = TavilySearchResults(max_results=5)

# LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.

    Summarize the following news into clear bullet points:

    {news}
    """
)

# Chain
chain = prompt | llm | StrOutputParser()

# Run search
news_result = search_tool.run("Latest AI news 2026")

# Invoke chain
result = chain.invoke({"news": news_result})

# Output
print(result)
