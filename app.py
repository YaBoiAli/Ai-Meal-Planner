import os
import re
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime, timedelta

load_dotenv()

memory = SqliteSaver.from_conn_string(":memory:")

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int
    ics_file: str

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)

PLAN_PROMPT = """You are an expert meal planner tasked with writing a meal plan. 
Please create a comprehensive meal plan based on the user's request. 
The plan should include an outline, relevant notes, detailed recipes, a shopping list, and instructions. 
Ensure the plan considers the user's dietary preferences and restrictions."""

WRITER_PROMPT = """You are a meal planner assistant tasked with creating an excellent meal plan based on the user's request and the initial outline provided. 
Please include the following for each day: 
-Breakfast, 
-Lunch, 
-Dinner. Also include snacks if applicable.
Generate a detailed shopping list for all ingredients needed for the meal plan. If the user provides feedback, 
revise the plan accordingly using the information below but still follow the outline:
------
{content}"""

REFLECTION_PROMPT = """You are an excellent critic reviewing a meal plan. 
Generate detailed critique and recommendations for the user's submission. 
Ensure the recipes meet the user's nutritional requirements, and dietary restrictions."""

RESEARCH_PLAN_PROMPT = """You are a researcher tasked with providing information to be used in writing a detailed meal plan. 
Generate a list of search queries to gather relevant information. Generate a maximum of 3 queries."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information for making requested revisions. 
Generate a list of search queries to gather relevant information. Generate a maximum of 3 queries."""

class Queries(BaseModel):
    queries: List[str]

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def parse_queries(response_content):
    pattern = r'\*\*"(.*?)"\*\*'
    queries = re.findall(pattern, response_content)
    return queries

def research_meal_plan_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    queries = parse_queries(response.content)
    content = state['content'] or []
    for q in queries:
        search_response = tavily.search(query=q, max_results=2)
        for r in search_response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my meal plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
    ]
    response = model.invoke(messages)
    
    state['draft'] = response.content  # Set 'draft' in the state

    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    if 'draft' not in state or not state['draft']:
        raise ValueError("No draft available for reflection.")
    
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ]
    response = model.invoke(messages)
    queries = parse_queries(response.content)
    content = state['content'] or []
    for q in queries:
        search_response = tavily.search(query=q, max_results=2)
        for r in search_response['results']:
            content.append(r['content'])
    return {"content": content}


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect_plan"

builder = StateGraph(AgentState)

builder.add_node("meal_planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect_plan", reflection_node)
builder.add_node("research_meal_plan", research_meal_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("meal_planner")

builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect_plan": "reflect_plan"}
)

builder.add_edge("meal_planner", "research_meal_plan")
builder.add_edge("research_meal_plan", "generate")
builder.add_edge("reflect_plan", "research_critique")
builder.add_edge("research_critique", "generate")
builder.add_edge("generate", "generate_ics")
builder.add_edge("generate_ics", END)

graph = builder.compile(checkpointer=memory)

def run_agent(task):
    thread = {"configurable": {"thread_id": "1"}}
    last_generate_state = {}
    for state in graph.stream({
        'task': task,
        "max_revisions": 2,
        "revision_number": 1
    }, thread):
        print(state, "\n\n")
        if state.get('draft'):
            last_generate_state['draft'] = state['draft']
        if state.get('ics_file'):
            last_generate_state['ics_file'] = state['ics_file']
    return last_generate_state

# Streamlit frontend
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

st.set_page_config(page_title="Meal Planner", page_icon="🍽️")
st.title("AI-Powered Meal Planner 🍽️ 1")
st.markdown("Create personalized meal plans with AI. Please provide your dietary preferences and restrictions, and our AI will generate a comprehensive meal plan for you.")

if "messages" not in st.session_state:
    st.session_state["messages"] = {}

def display_messages():
    for name, content in st.session_state.messages.items():
        st.chat_message(name).write(content)

task = st.text_area("Enter your Meal plan")
if st.button("Generate Meal Plan"):
    if task:
        try:
            with st.spinner("Generating your meal plan..."):
                response = run_agent(task)
            if response:
                st.session_state.messages[ROLE_ASSISTANT] = response['draft']
                st.session_state["ics_file"] = response.get("ics_file", "")
            display_messages()
            st.markdown(response['draft'])
            if "ics_file" in st.session_state and st.session_state["ics_file"]:
                st.markdown(f"[Download ICS file](./meal_plan.ics)")
        except Exception as e:
            st.error(f"Error: {e}")

display_messages()
