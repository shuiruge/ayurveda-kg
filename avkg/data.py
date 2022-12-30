import json
import os

from kg import SUBCATEGORY_RELATION, Entity, Fact, KnowledgeGraph, Relation


def update_knowledge_graph(kg: KnowledgeGraph,
                           data_path: str,
                           category: str = None):
    """Inplace update the knowledge graph `kg` by the data in the
    path `data_path`.

    Parameters
    ----------
    data_path: The path to the JSON file of the pre-structured data.
        We store the data in a specific JSON format, wherein the keys
        are the heads, and values are also JSON, but with the keys
        relations and the values tails or list of tails.

        For example, in "data/taste.json",

            {
                "sweet": {
                    "component": [
                        "earth",
                        "water"
                    ]
                },
                "sour": {
                    "component": [
                        "earth",
                        "fire"
                    ]
                },
                ...
            }

    kg: The base knowledge graph, on which new data are added.
    """
    with open(data_path, 'r') as f:
        heads = json.load(f)

    for head, relations in heads.items():
        kg.add(Fact(head, SUBCATEGORY_RELATION, category))
        for relation, tails in relations.items():
            tails = [tails] if isinstance(tails, str) else tails
            for tail in tails:
                kg.add(Fact(head, relation, tail))


def _load_data_recur(data_dir_path: str,
                     kg: KnowledgeGraph = None):
    if kg is None:
        kg = KnowledgeGraph()
        category = None
    else:
        _, category = os.path.split(data_dir_path)

    for filename in os.listdir(data_dir_path):
        file_path = os.path.join(data_dir_path, filename)

        if os.path.isdir(file_path):
            kg = _load_data_recur(file_path, kg)

        else:
            subcategory, ext = os.path.splitext(filename)
            assert ext == '.json'
            if category is not None:
                kg.add(Fact(subcategory, SUBCATEGORY_RELATION, category))
            update_knowledge_graph(kg, file_path, subcategory)
    return kg


def load_data(data_dir_path: str):
    return _load_data_recur(data_dir_path, None)
