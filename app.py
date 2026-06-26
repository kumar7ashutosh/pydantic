import os,streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain.agents import create_agent
import requests
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
WEATHER_STACK_API_KEY = os.environ.get("WEATHER_STACK_API_KEY")

st.set_page_config(page_title="LangChain AI Agent", page_icon="🤖", layout="centered")
st.title("LangChain AI Agent")
st.markdown("This is a simple AI agent that can use tools to answer questions. It uses the LangChain library and OpenAI's GPT-4 model.")
search_tool=TavilySearchResults(max_results=3,)

@tool
def get_weather_data(city:str)->str:
    """Get the current weather for a given city."""
    url=(
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHER_STACK_API_KEY}&query={city}"
        )
    response=requests.get(url)
    data=response.json()
    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

llm=ChatOpenAI(model=os.getenv("MODEL_NAME"),temperature=0,openai_api_key=OPENAI_API_KEY)
tools = [
    search_tool,
    get_weather_data
]
prompt="""You are a helpful assistant that can use tools.

Think step by step.
Use tools whenever necessary.
After observing tool results, provide the final answer.
"""
agent=create_agent(model=llm,tools=tools,system_prompt=prompt)
user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)
if st.button("Run Agent"):
    if user_query:
        with st.spinner("Running agent..."):
            response=agent.invoke({
                "messages":[("user",user_query)]
            })
            st.success("Agent finished running.")
            st.markdown("### Agent Response:")
            st.write(response["messages"][-1].content)
    else:
        st.warning("Please enter a query before running the agent.")