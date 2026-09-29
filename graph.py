import json
import random
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from extract import client
from embedding import get_collection, CHUNKS_AMOUNT

QUESTIONS = [
    "How did the band`s record label change over the course of their discography and what does this suggest about their trajectory?",
    "What recurring themes appear across the band`s album concepts, from I Love You. to Ultrasounds?"
]

def load_data(file_name):
    path = Path(file_name)
    return json.loads(path.read_text(encoding="utf-8"))

def build_graph(bands):
    graph = nx.Graph()
    for band in bands:
        graph.add_node(band["name"], type="band")
        graph.add_node(band["yearCreated"], type="year")
        graph.add_edge(band["name"], band["yearCreated"], relation="created_in")

        for album in band["albums"]:
            graph.add_node(album["name"], type="album")
            graph.add_edge(band["name"], album["name"], relation="released")

            for song in album["songs"]:
                graph.add_node(song["name"], type="song")
                graph.add_edge(album["name"], song["name"], relation="contains")

            for producer in album["producers"]:
                graph.add_node(producer["name"], type="person")
                graph.add_edge(album["name"], producer["name"], relation="produced_by")
            for review in album["reviews"]:
                graph.add_node(review["publicationName"], type="review")
                graph.add_edge(album["name"], review["publicationName"], relation="reviewed")
                critic = review["critic"]
                if isinstance(critic, dict):
                    graph.add_node(critic["name"], type="person")
                    graph.add_edge(album["name"], critic["name"], relation="reviewed_by")

        for member in band["members"]:
            graph.add_node(member["name"], type="person")
            graph.add_edge(band["name"], member["name"], relation="is_part_of")
        for event in band["events"]:
            graph.add_node(event["name"], type="event")
            graph.add_edge(band["name"], event["name"], relation="involved_in")

    return graph

def visualize(graph, save_path):
    color_list = {
        "band": "#e74c3c",
        "album": "#f39c12",
        "song": "#85c1e9",
        "person": "#82e0aa",
        "review": "#bb8fce",
        "event": "#aeb6bf"}
    colors = [color_list.get(graph.nodes[n].get("type"), "pink") for n in graph.nodes]
    pos = nx.spring_layout(graph, k=0.6, seed=100)
    nx.draw(graph, pos, with_labels=True, node_size=300, font_size=4, edge_color='gray', width=0.5, node_color=colors)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_graph_communities(g):
    communities = community_louvain.best_partition(g, resolution=0.1)
    print("FOUNDED COMMUNITIES:")
    print(f"|{'-' * 52}|{'-' * 18}|")
    print(f"|{'NODE':^52}|{'COMMUNITY':^18}|")
    print(f"|{'-' * 52}|{'-' * 18}|")

    groups = {}
    for node, community in communities.items():
        print(f"|{node:^51} | {community:^17}|")
        print(f"|{'-' * 52}|{'-' * 18}|")
        groups.setdefault(community, []).append(node)

    return {key: v for key, v in (sorted(groups.items(), key=lambda  item: item[0]))}

def show_community_data(community):
    for community, nodes in community.items():
        print(f"{community}: {nodes}")

def model(prompt):
    return client.models.generate_content(
        model="gemini-3.1-flash-lite",
        # model="gemini-3.7-flash",
        contents=prompt
    )

def make_request_by_community(entities, relationships, community_number):
    prompt = f"""
            Pleas analyze a community of knowledge graph about the music band The Neighbourhood.
            Community №{community_number}
            Entities in the community: {entities}
            Relationships in the community: {relationships}
            Based only on the provided entities and relationships, write a short summary of the community. 
            """

    interaction = model(prompt)
    print(f"COMMUNITY №{community_number} RESPONSE:\n{interaction.text}\n")
    return interaction.text

def generate_summary(prompts):
    prompt = f"""
    Give me summarizes of the communities detected in the graph: {prompts}
    Based on these information, write a final concise overview of the knowledge graph.
    Describe the main entities, relationships and topic represented across the community.
    """

    interaction = model(prompt)
    print(f"GENERAL SUMMARY:\n{interaction.text}\n")
    return interaction.text

def resume(groups, graph):
    results = []
    for community, nodes in groups.items():
        entities = []
        relationships = []
        for node in nodes:
            entities.append({
                "node": node,
                "type": graph.nodes[node]["type"]
            })

        unique_nodes = set(nodes)
        for node_from, node_to, data in graph.edges(data=True):
            if node_from in unique_nodes and node_to in unique_nodes:
                relationships.append({
                    "from": node_from,
                    "to": node_to,
                    "relation": data.get("relation")
                })
        results.append(make_request_by_community(entities, relationships, community))

    summary = "\n".join(results)
    # generate_summary(summary)
    return summary

def get_answer_by_graph(question, summary):
    prompt = f"""
    Answer the following question about the band The Neighbourhood: {question}
    The following information consists of summaries of communities
    detected in the knowledge graph: {summary}
    Using only this information, provide a complete answer to the question.
    Connect information from different communities and identify
    relationships or general patterns that emerge across them.
    Do not add information that is not present in the provided summaries.
    """
    interaction = model(prompt)
    return interaction.text

def get_answer_by_chunks(question):
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=CHUNKS_AMOUNT)
    chunks = results["documents"][0]
    context = "\n".join(chunks)
    prompt = f"""
    Answer the following question about the band The Neighbourhood: {question}.
    Use only the information from the retrieved chunks: {context}.
    Provide a complete answer.
    Do not add information that is not present in the chunks.
    """
    interaction = model(prompt)
    return interaction.text


if __name__ == "__main__":
    bands = load_data("facts.json")
    graph = build_graph(bands)
    # print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    visualize(graph, "graph.png")
    print("GRAPH CREATED...")
    # for u, v, data in graph.edges(data=True):
    #     print(u, "->", v, data)

    groups = get_graph_communities(graph)
    show_community_data(groups)
    summary = resume(groups, graph)

    selected_question = QUESTIONS[random.randint(0, len(QUESTIONS) - 1)]
    graphs_answer = get_answer_by_graph(selected_question, summary)
    print(f"QUESTION:\n{selected_question}")
    print(f"\nANSWER (GRAPH-RAG)\n: {graphs_answer}")
    chunks_answer = get_answer_by_chunks(selected_question)
    print(f"\nANSWER (VECTOR-RAG)\n: {chunks_answer}")
