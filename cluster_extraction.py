import os
import json
from typing import Dict, List
from functools import reduce

import networkx as nx
from dotenv import load_dotenv
from sklearn.cluster import SpectralClustering
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field, BaseModel


load_dotenv()


class NodeIds(BaseModel):
    node_ids: List[str] = Field(description="IDs of nodes related to the user's query")


def detect_communities(G: nx.Graph, n_clusters: int = 5) -> Dict[int, List[int]]:
    """
    Detect communities in a given graph using spectral clustering.

    Args:
        G (nx.Graph): The input graph on which to detect communities.
        n_clusters (int, optional): The number of clusters to form. Defaults to 5.

    Returns:
        Dict[int, List[int]]: A dictionary mapping each cluster index to a list of node IDs 
        belonging to that cluster.
    """
    adj_matrix = nx.to_numpy_array(G)
    clustered = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed"
    ).fit_predict(adj_matrix)
    communities = {}
    for node, label in zip(G.nodes, clustered):
        communities.setdefault(int(label), []).append(node)
    return communities


def map_phase(query: str) -> callable:
    """
    Create a function that maps a list of graph nodes to a JSON object containing
    relevant node IDs based on a user query.

    The generated function, `map_func`, takes in a string representation of nodes and 
    returns a JSON string with a list of node IDs related to the query. It leverages a 
    language model to interpret the query and select relevant nodes.

    Args:
        query (str): The user query or coding task to base the node selection on.

    Returns:
        callable: A function that takes in a string of nodes and returns a JSON string 
        with a list of relevant node IDs.
    """

    def map_func(nodes: str) -> str:
        template = """
        # BPMN Assistant Prompt

        You are a BPMN expert and assistant on a low-code platform.  

        1. Select the graph nodes that are most relevant to the given user query or coding task.  
        2. Return the result as a JSON object containing a list of node IDs.  

        ## Query
        {query}
        
        ## Nodes
        {nodes}
        """
        llm = ChatOpenAI(
            temperature=0.0,
            model=os.environ.get("BIELIK_MODEL"),
            base_url=os.environ.get("BIELIK_BASE_URL"),
            api_key=os.environ.get("BIELIK_API_KEY"),
        )
        prompt = PromptTemplate.from_template(template)
        parser = JsonOutputParser(pydantic_object=NodeIds)
        chain = prompt | llm | parser
        return chain.invoke(input={"query": query, "nodes": nodes})

    return map_func


def reduce_phase(G: nx.Graph) -> callable:
    """
    Create a function that aggregates a list of JSON objects containing node IDs
    and their associated data.

    The generated function, `reduce_func`, takes in a string and a dictionary with
    a list of node IDs and returns a concatenated string of the JSON objects
    representing the nodes.

    Args:
        G (nx.Graph): The input graph on which the nodes exist.

    Returns:
        callable: A function that takes in a string and a dictionary with a list of
        node IDs and returns a concatenated string of the JSON objects representing
        the nodes.
    """
    def reduce_func(accumulator: str, related_nodes: dict) -> str:
        for node_id in related_nodes["nodeIds"]:
            print(json.dumps(G.nodes[node_id]))
            accumulator += json.dumps(G.nodes[node_id])
        return accumulator

    return reduce_func


def prepare_summaries(G: nx.Graph, communities: Dict[int, List[int]]) -> List[str]:
    """
    Prepare summaries for each community in the graph.

    This function takes in a graph and a dictionary of communities, where each entry
    consists of a cluster index and a list of node IDs belonging to that cluster.
    It generates a summary for each community based on the node details. The
    summaries are concatenated strings of the JSON objects representing the
    nodes.

    Args:
        G (nx.Graph): The input graph containing node information.
        communities (Dict[int, List[int]]): A dictionary mapping each cluster index
            to a list of node IDs belonging to that cluster.

    Returns:
        List[str]: A list of summaries for each community.
    """
    summaries = []
    for _, nodes in communities.items():
        node_details = [json.dumps(G.nodes[node]) for node in nodes]
        summaries.append("\n".join(node_details))
    return summaries


def mapreduce(G: nx.Graph, query: str, communities: Dict[int, List[int]]) -> str:
    """
    Maps a list of summaries to a list of answers, then reduces the list of answers to a single answer.

    The mapping function takes in a string summary and returns a string answer.
    The reducing function takes in two string answers and returns a single string answer.

    Parameters:
        G (nx.Graph): The input graph on which the nodes exist.
        query (str): The user query or coding task to base the node selection on.
        communities (Dict[int, List[int]]): A dictionary mapping each cluster index
            to a list of node IDs belonging to that cluster.

    Returns:
        str: A single string answer.
    """
    summaries = prepare_summaries(G, communities)
    nodes = map(map_phase(query), summaries)
    context = reduce(reduce_phase(G), nodes, "")
    return context


def main() -> None:
    with open("bpmn.json", "r") as f:
        data = json.load(f)

        G = nx.Graph()
        for node in data["nodes"]:
            G.add_node(node["id"], **node)

        for node in data["nodes"]:
            for edge in node["edges"]:
                G.add_edge(node["id"], edge["to"], **edge)

    communities = detect_communities(G)
    query = "filter nextTasks by user input"
    print("===================")
    answer = mapreduce(G, query, communities)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
