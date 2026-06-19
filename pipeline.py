from langgraph.graph import StateGraph, END
from state import PaperState
from reader import reader_node
from chunker import chunker_node
from extractor import extractor_node
from github_checker import github_checker_node
from inspector import inspector_node


def build_pipeline():
    graph = StateGraph(PaperState)

    graph.add_node("reader", reader_node)
    graph.add_node("chunker", chunker_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("github_checker", github_checker_node)
    graph.add_node("inspector", inspector_node)

    graph.add_edge("reader", "chunker")
    graph.add_edge("chunker", "extractor")
    graph.add_edge("extractor", "github_checker")
    graph.add_edge("github_checker", "inspector")
    graph.add_edge("inspector", END)

    graph.set_entry_point("reader")

    return graph.compile()


pipeline = build_pipeline()