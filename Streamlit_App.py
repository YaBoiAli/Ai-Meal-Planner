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
import sys
import io
import streamlit as st

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

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4)

PLAN_PROMPT = """You are an expert meal outline planner tasked with creating a 7-day meal plan outline.
Give the outline of the meal plan along with any relevant notes, calories,
recipes based on user preferences, shopping list based on ingredients, available ingredients or instructions for the recipe."""

WRITER_PROMPT = """You are an excellent meal planner generator tasked with writing an excellent 7-day meal plan with schedules.
Write a detailed and concise final 7-day meal plan Following this template:
Day 1:
  Breakfast -
  Lunch -
  Dinner -
Day 2:
  Breakfast -
  Lunch -
  Dinner -
...
Day 7:
  Breakfast -
  Lunch -
  Dinner -
Add optional snacks in between these times.
Please include the shopping list, calories, protein, and ingredients for the meal plan.
Generate the best meal plan possible for the user's request based on the provided template,
Provide every detail concisely.
If the user provides critique, respond with a revised version of your previous attempts.
Use all the information below as needed:
------
{content}"""

REFLECTION_PROMPT = """You are a critic reviewing a meal plan. 
Generate critique and recommendations for the user's meal plan. 
Select the best recipes considering nutritional requirements and dietary restrictions. 
Filter the recipes to ensure they meet the user's nutritional requirements and dietary restrictions considering calories and protein."""

RESEARCH_PLAN_PROMPT = """You are a researcher tasked with providing information to be used in writing a detailed meal plan according to the user meal plan outline. 
Generate a list of search queries to gather relevant information regarding calories, protein, ingredients, and recipes. Generate a maximum of 3 queries."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can 
be used when making any requested revisions. Generate a list of search queries to gather relevant information. Generate a maximum of 3 queries."""

class Queries(BaseModel):
    queries: List[str]

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def ensure_7_day_plan(task):
    if "7-day meal plan" not in task:
        task += "\nPlease create a 7-day meal plan."
    return task

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    plan_node_result = print("Plan agent Response: ", response.content)
    plan_node_result
    return {"plan": response.content}

def parse_queries(response_content):
    # Extract queries from the response content using regex
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
    research_meal_plan_node = print("Research Meal Plan Response:", response.content)  # Debug print
    research_meal_plan_node
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
    generation_node = print("Generation Response: ", response.content)
    generation_node
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    reflection_node = print("Reflection Response:", response.content)
    reflection_node
    return {"critique": response.content}

def ensure_7_day_plan(task):
    if "7-day meal plan" not in task:
        task += "\nPlease create a 7-day meal plan."
    return task


def research_critique_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ]
    response = model.invoke(messages)
    research_critique_node = print("Research Critique Response:", response.content)  # Debug print
    research_critique_node
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

graph = builder.compile(checkpointer=memory)

def start_agents(task):
    # Save the current stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to a buffer
    
    try:
        thread = {"configurable": {"thread_id": "1"}}
        responses = list(graph.stream({
            'task': task,
            "max_revisions": 2,
            "revision_number": 1
        }, thread))
    finally:
        # Restore the original stdout
        sys.stdout = old_stdout
    
    if responses:
        draft = responses[-1].get('generate', {}).get('draft', 'No draft found')
        return draft
    else:
        "No responses received"

# Streamlit page configuration
st.set_page_config(page_title="Meal Planner", page_icon="🍽️")
st.title("AI-Powered Meal Planner 🍽️")
st.markdown("Create personalized meal plans with AI. Please provide your dietary preferences and restrictions, and our AI will generate a comprehensive meal plan for you.")

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Text area for meal plan input
task = st.text_area("Enter your Meal Plan")

# Button to trigger meal plan generation
if st.button("Generate Meal Plan"):
    if task:
        task = ensure_7_day_plan(task)  # Ensure it requests a 7-day plan
        try:
            with st.spinner("Generating your meal plan..."):
                draft = start_agents(task)
            st.subheader("Generated Meal Plan:")
            st.markdown(draft)
        except Exception as e:
            st.error(f"Error: {e}")