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
import gradio as gr
import json

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

PLAN_PROMPT = """You are an expert meal planner tasked with writing a meal plan. \
Write a meal plan for the user provided topic. Give an outline of the meal plan along with any relevant notes, \
recipes based on user preferences, shopping list based on ingredients, available ingredients or instructions for the sections."""

WRITER_PROMPT = """You are a meal planner assistant tasked with writing excellent meal plans. \
Generate the best meal plan possible for the user's request and the initial outline. \
Follow this outline: Day 1: -Breakfast, -Lunch, -Dinner with the specific time. \
Do include the shopping list for the meal plan. \
If the user provides critique, respond with a revised version of your previous attempts. \
Use all the information below as needed: 
------ 
{content}"""

REFLECTION_PROMPT = """You are a critic reviewing a meal plan. \
Generate critique and recommendations for the user's submission. \
Select the best recipes considering nutritional requirements and dietary restrictions. \
Filter the recipes to ensure they meet the user's nutritional requirements and dietary restrictions."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following meal plan. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

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
    print("Meal Plan Response:", response.content)  # Debug print
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
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ]
    response = model.invoke(messages)
    print("Research Critique Response:", response.content)  # Debug print
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

def meal_planner_interface(task, max_revisions):
    state = {
        'task': task,
        'max_revisions': max_revisions,
        'revision_number': 1
    }
    thread = {"configurable": {"thread_id": "1"}}
    outputs = []
    for s in graph.stream(state, thread):
        outputs.append(s)
    return "\n\n".join([str(output['draft']) for output in outputs])

interface = gr.Interface(
    fn=meal_planner_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your meal planning task..."),
        gr.Slider(1, 5, step=1, label="Max Revisions")
    ],
    outputs="text",
    title="AI Meal Planner",
    description="Generate meal plans based on your dietary requirements and preferences."
)

interface.launch()