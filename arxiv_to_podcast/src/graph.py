import asyncio
import os
from operator import add
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from src.templates import enhance_prompt, initial_dialogue_prompt, plan_prompt
from src.utils.audio import PodcastGenerator
from src.utils.utilities import (
    download_arxiv_pdf,
    get_discussion_chain,
    get_paper_head,
    initialize_discussion_chain,
    initialize_vectorstore,
    parse_arxiv_pdf,
    parse_script_plan,
)

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

    section_plan: List[str]
    script: Annotated[str, add]
    enhanced_script: str
    audio_path: str


def process_arxiv_paper(state: PodcastState) -> Dict[str, Any]:
    paper_url = state["paper_url"]

    pdf_path, paper_id = download_arxiv_pdf(paper_url)
    parsed_path = parse_arxiv_pdf(pdf_path)

    with open(parsed_path, encoding="utf-8") as f:
        paper_content = f.read()

    return {
        "paper_id": paper_id,
        "paper_raw_path": pdf_path,
        "paper_parsed_path": parsed_path,
        "paper_content": paper_content,
    }


def generate_script_plan(state: PodcastState) -> Dict[str, Any]:
    response = llm.invoke(
        [HumanMessage(content=plan_prompt.format(paper=state["paper_content"]))]
    )
    plan = parse_script_plan(response)
    return {"section_plan": plan}


def generate_initial_dialogue(state: PodcastState) -> Dict[str, Any]:
    paper_head = get_paper_head(state["paper_raw_path"])
    response = llm.invoke(
        [HumanMessage(content=initial_dialogue_prompt.format(paper_head=paper_head))]
    )
    return {"script": response.content}


def process_section(state: PodcastState) -> Dict[str, Any]:
    discuss_chain = initialize_discussion_chain(state["paper_parsed_path"], llm)

    current_section = state["section_plan"][0]
    section_script = discuss_chain.invoke(
        {"section_plan": current_section, "previous_dialogue": state["script"]}
    )
    remaining_sections = state["section_plan"][1:]
    return {"script": section_script, "section_plan": remaining_sections}


def should_continue_sections(state: PodcastState) -> Dict[str, Any]:
    if state["section_plan"]:
        return "enhance_script"
    return "enhance_script"


def enhance_script(state: PodcastState) -> Dict[str, Any]:
    response = llm.invoke(
        [HumanMessage(content=enhance_prompt.format(draft_script=state["script"]))]
    )
    enhanced_script = response.content
    paper_dir = os.path.dirname(state["paper_raw_path"])
    enhanced_script_path = os.path.join(paper_dir, "enhanced_script.txt")

    with open(enhanced_script_path, "w", encoding="utf-8") as f:
        f.write(enhanced_script)

    return {"enhanced_script": enhanced_script}


def generate_podcast(state: PodcastState) -> Dict[str, Any]:
    paper_dir = os.path.dirname(state["paper_raw_path"])
    generator = PodcastGenerator(
        subscription_key=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
        base_dir=paper_dir,
    )
    audio_path = asyncio.run(generator.generate_podcast(state["enhanced_script"]))
    return {"audio_path": audio_path}


def create_arxiv_to_podcast_agent(verbose: bool = True):
    graph = StateGraph(PodcastState)

    def add_verbose_node(name, func):
        def verbose_wrapper(state):
            if verbose:
                print("\n" + "=" * 50)
                print(f"🔄 Executing: {name}")

                # Show relevant state based on node type
                if name == "process_section":
                    total_sections = len(state.get("section_plan", []))
                    remaining = len(state["section_plan"])
                    completed = total_sections - remaining
                    print(f"📊 Progress: Section {completed}/{total_sections}")
                    if state["section_plan"]:
                        print(f"🎯 Current section: {state['section_plan'][0][:100]}...")
                elif name == "process_arxiv_paper":
                    print(f"📄 Processing paper: {state['paper_url']}")
                    # memory.vectorstore = initialize_vectorstore(state["paper_parsed_path"])
                elif name == "generate_script_plan":
                    print("📝 Generating podcast structure...")
                elif name == "generate_podcast":
                    print("🎙️ Generating podcast audio...")

            result = func(state)

            if verbose:
                print(f"✅ Completed: {name}")
                if name == "generate_podcast":
                    print(f"📁 Audio saved to: {result.get('audio_path')}")
                print("=" * 50 + "\n")
            return result

        graph.add_node(name, verbose_wrapper)

    add_verbose_node("process_arxiv_paper", process_arxiv_paper)
    add_verbose_node("generate_script_plan", generate_script_plan)
    add_verbose_node("generate_initial_dialogue", generate_initial_dialogue)
    add_verbose_node("process_section", process_section)
    add_verbose_node("enhance_script", enhance_script)
    # add_verbose_node("generate_podcast", generate_podcast)

    graph.add_edge("process_arxiv_paper", "generate_script_plan")
    graph.add_edge("process_arxiv_paper", "generate_initial_dialogue")
    graph.add_edge("generate_initial_dialogue", "process_section")

    # Add conditional edge for section processing
    graph.add_conditional_edges(
        "process_section",
        should_continue_sections,
        {"process_section": "process_section", "enhance_script": "enhance_script"},
    )
    graph.add_edge("enhance_script", END)
    graph.add_edge("generate_script_plan", END)
    # graph.add_edge("generate_podcast", END)

    graph.set_entry_point("process_arxiv_paper")

    return graph.compile()
