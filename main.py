from src.graph import create_arxiv_to_podcast_agent

url = "https://arxiv.org/pdf/2103.00020.pdf"
graph = create_arxiv_to_podcast_agent()

initial_state = {"paper_url": url}
result = graph.invoke(initial_state)
print(result)

