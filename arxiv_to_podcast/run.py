#!/usr/bin/env python

import logging

from dotenv import load_dotenv
from naptha_sdk.client.naptha import Naptha
from naptha_sdk.configs import load_agent_deployments
from naptha_sdk.schemas import AgentRunInput
from src.graph import create_arxiv_to_podcast_agent

from arxiv_to_podcast.schemas import InputSchema

load_dotenv()
logger = logging.getLogger(__name__)


class ArxivToPodcastAgent:
    def __init__(self, verbose: bool = True):
        self.graph = create_arxiv_to_podcast_agent(verbose=verbose)

    def execute(self, arxiv_url: str) -> dict:
        logger.info(f"Processing arxiv paper: {arxiv_url}")

        # Execute graph with initial state
        initial_state = {"paper_url": arxiv_url}
        result = self.graph.invoke(initial_state)

        return result


def run(agent_run: AgentRunInput, *args, **kwargs):
    agent = ArxivToPodcastAgent()
    result = agent.execute(agent_run.inputs.arxiv_url)
    return result


if __name__ == "__main__":
    naptha = Naptha()

    agent_deployments = load_agent_deployments(
        "arxiv_to_podcast/configs/agent_deployments.json",
        load_persona_data=False,
        load_persona_schema=False,
    )
    inputs = InputSchema(arxiv_url="https://arxiv.org/pdf/2103.00020.pdf")

    agent_run = AgentRunInput(
        inputs=inputs,
        agent_deployment=agent_deployments[0],
        consumer_id=naptha.user.id,
    )

    print(run(agent_run))
