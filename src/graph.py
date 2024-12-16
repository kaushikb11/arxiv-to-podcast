import os
from typing import Any, Dict, List, TypedDict, Annotated
from dotenv import load_dotenv
from operator import add

from langgraph.graph import END, StateGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from src.utils.utilities import download_arxiv_pdf, parse_arxiv_pdf, parse_script_plan, get_paper_head
from src.templates import plan_prompt, initial_dialogue_prompt, enhance_prompt

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

class PodcastState(TypedDict):
    paper_url: str
    paper_content: str
    paper_id: str
    paper_raw_path: str
    paper_parsed_path: str
    
    section_script_plan: List[str]
    script: Annotated[str, add] 
    
    
def process_arxiv_paper(state: PodcastState) -> Dict[str, Any]:
    paper_url = state["paper_url"]
    
    pdf_path, paper_id = download_arxiv_pdf(paper_url)
    parsed_path = parse_arxiv_pdf(pdf_path)
    
    with open(parsed_path, 'r', encoding='utf-8') as f:
        paper_content = f.read()
    
    return {
        "paper_id": paper_id,
        "paper_raw_path": pdf_path,
        "paper_parsed_path": parsed_path,
        "paper_content": paper_content
    }
    
def generate_script_plan(state: PodcastState) -> Dict[str, Any]:
    response = llm.invoke([HumanMessage(content=plan_prompt.format(paper=state["paper_content"]))])
    plan = parse_script_plan(response)
    return {"section_script_plan": plan}

def generate_initial_dialogue(state: PodcastState) -> Dict[str, Any]:
    paper_head = get_paper_head(state["paper_raw_path"])
    response = llm.invoke([HumanMessage(content=initial_dialogue_prompt.format(paper_head=paper_head))])
    return {"script": response.content}

def process_section(state: PodcastState) -> Dict[str, Any]:
    # TODO
    return {}

def should_continue_sections(state: PodcastState) -> Dict[str, Any]:
    if state["section_plan"]:
        return "process_section"
    return "enhance"

def enhance_script(state: PodcastState) -> Dict[str, Any]:
    # TODO
    return {}

def create_arxiv_to_podcast_pdf(verbose: bool = True):
    
    graph = StateGraph(PodcastState)

    def add_verbose_node(name, func):
        def verbose_wrapper(state):
            if verbose:
                print(f"Starting {name}...")
            result = func(state)
            if verbose:
                print(f"Finished {name}")
            return result

        graph.add_node(name, verbose_wrapper)
        
    add_verbose_node("process_arxiv_paper", process_arxiv_paper)
    add_verbose_node("generate_script_plan", generate_script_plan)
    add_verbose_node("generate_initial_dialogue", generate_initial_dialogue)
    
    graph.add_edge("process_arxiv_paper", "generate_script_plan")
    graph.add_edge("process_arxiv_paper", "generate_initial_dialogue")
    graph.add_edge("generate_script_plan", END)
    graph.add_edge("generate_initial_dialogue", END)
    
    graph.set_entry_point("process_arxiv_paper")
        
    return graph.compile()