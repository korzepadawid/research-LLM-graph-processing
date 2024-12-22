import json
from typing import Dict, List
from functools import reduce

import networkx as nx
from dotenv import load_dotenv
from sklearn.cluster import SpectralClustering
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def detect_communities(G: nx.Graph, n_clusters: int = 3) -> Dict[int, List[int]]:
    """
    Detect communities in a given graph using spectral clustering.

    Args:
        G (nx.Graph): The input graph on which to detect communities.
        n_clusters (int, optional): The number of clusters to form. Defaults to 3.

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


def write_community_summary(node_details: List[str]) -> str:
    """
    Summarize a list of node details into a single string.

    This function takes in a list of strings, where each string represents a node
    in the graph. It then uses the gpt-4 model to generate a summary of the nodes,
    which is a single string that includes all the details of the nodes in a
    concise manner.

    Args:
        node_details (List[str]): A list of strings, where each string represents
            a node in the graph.

    Returns:
        str: A single string summary of the nodes.
    """
    template = """
    Write a summary of the following nodes, make sure to include all the details:
    {details}
    Don't write anything else.
    """
    details = "\n".join(node_details)
    llm = ChatOpenAI(temperature=0.0, model="gpt-4")
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(input={"details": details})


def summary_communities(
    G: nx.Graph, communities: Dict[int, List[int]], llm_summary: bool = False
) -> List[str]:
    """
    Generate summaries for communities within a graph.

    This function iterates over a dictionary of communities, where each 
    entry consists of a cluster index and a list of node IDs belonging to 
    that cluster. It generates a summary for each community based on the 
    node details. If `llm_summary` is set to True, it uses a language model 
    to create a concise summary of the node details. Otherwise, it 
    concatenates the node details into a single string.

    Args:
        G (nx.Graph): The input graph containing node information.
        communities (Dict[int, List[int]]): A dictionary mapping each cluster 
            index to a list of node IDs belonging to that cluster.
        llm_summary (bool, optional): Whether to use a language model to 
            generate a concise summary of the node details. Defaults to False.

    Returns:
        List[str]: A list of summaries for each community.
    """

    summaries = []
    for _, nodes in communities.items():
        node_details = [json.dumps(G.nodes[node]) for node in nodes]
        if llm_summary:
            summaries.append(write_community_summary(node_details))
        else:
            summaries.append("\n".join(node_details))
    return summaries


def map_phase(query: str) -> callable:
    """
    Create a function that maps a summary to an answer to a given query.

    The generated function takes in a string summary and returns a string answer.
    It uses the gpt-4 model to generate the answer based on the summary.

    Parameters:
        query (str): The query to answer.

    Returns:
        callable: A function that takes in a string summary and returns a string answer.
    """
    def map_func(summary: str) -> str:
        template = """
        Please answer the following query based on the provided summary. If the summary does not contain the answer, respond with "I don't know."
        <query>
            {query}
        </query>
        <summary>
            {summary}
        </summary>
        """
        llm = ChatOpenAI(temperature=0.0, model="gpt-4")
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(input={"query": query, "summary": summary})

    return map_func


def reduce_phase(query: str) -> callable:
    """
    Create a function that reduces a list of answers to a single answer.

    The generated function takes in two string answers and returns a single string answer.
    It uses the gpt-4 model to generate the answer based on the two input answers.

    Parameters:
        query (str): The query to answer.

    Returns:
        callable: A function that takes in two string answers and returns a single string answer.
    """
    def reduce_func(accumulator: str, answer: str) -> str:
        template = """
        Please answer the following query based on the two provided answers.
        If either answer is sufficient on its own, feel free to use it. 
        Only combine the answers if needed for a more complete response. 
        Do not invent any information.
        <query>
            {query}
        </query>
        <answer>
            {accumulator}
        </answer>
        <answer>
            {answer}
        </answer>
        """

        llm = ChatOpenAI(temperature=0.0, model="gpt-4")
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            input={"query": query, "accumulator": accumulator, "answer": answer}
        )

    return reduce_func


def mapreduce(query: str, summaries: List[str]) -> str:
    """
    Maps a list of summaries to a list of answers, then reduces the list of 
    answers to a single answer.

    The mapping function takes in a string summary and returns a string answer.
    The reducing function takes in two string answers and returns a single string 
    answer.

    Parameters:
        query (str): The query to answer.
        summaries (List[str]): A list of summaries to map and reduce.

    Returns:
        str: A single string answer.
    """
    answers = map(map_phase(query), summaries)
    return reduce(reduce_phase(query), answers)


def main() -> None:
    with open("bpmn.json", "r") as f:
        data = json.load(f)

        G = nx.Graph()
        for node in data["nodes"]:
            print("Node:", node)
            G.add_node(node["id"], **node)

        for node in data["nodes"]:
            for edge in node["edges"]:
                G.add_edge(node["id"], edge["to"], **edge)

    communities = detect_communities(G)
    print("Communities:", communities)
    summaries = summary_communities(G, communities)
    print("Summaries:", summaries)
    query = input(str("Query: "))
    print("===================")
    answer = mapreduce(query, summaries)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
