import json
import os
from collections import defaultdict
from typing import Dict, Optional, Set, Union


class Entity:

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self.name}'

    def __str__(self):
        return self.name

    def is_like(self, name: str):
        # TODO: Temporal implementation without NLP.
        return self.name == name


class Relation:

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def is_like(self, name: str):
        # TODO: Temporal implementation without NLP.
        return self.name == name


class Fact:

    def __init__(self,
                 head: Union[Entity, str],
                 relation: Union[Relation, str],
                 tail: Union[Entity, str]):
        if isinstance(head, str):
            head = Entity(head)
        self.head = head

        if isinstance(relation, str):
            relation = Relation(relation)
        self.relation = relation

        if isinstance(tail, str):
            tail = Entity(tail)
        self.tail = tail

    def __repr__(self):
        return f'{repr(self.head)} -- {repr(self.relation)} --> {repr(self.tail)}'


class KnowledgeGraph:
    """The elemental data structure of knowledge graph.

    Based on ref[1], section II.B, knowledge graph is defined as
    triplet (E, R, F), where E for entities, R for relations, and
    F for facts. Fact is defined as triplet (h, r, t), where head
    h and tail t are in E and relation r in R.

    References
    ----------
    1. [A Survey on Knowledge Graphs: Representation, Acquisition and Applications](https://arxiv.org/abs/2002.00388v4).
    """

    SUBCATEGORY_RELATION = 'is of'

    def __init__(self):
        self._entities: Set[Entity] = set()
        self._relations: Set[Relation] = set()
        self._facts: Set[Fact] = set()

        self._head_to_facts: Dict[Entity, Set[Fact]] = defaultdict(set)
        self._relation_to_facts: Dict[Relation, Set[Fact]] = defaultdict(set)
        self._tail_to_facts: Dict[Entity, Set[Fact]] = defaultdict(set)

    @property
    def entities(self):
        return self._entities

    @property
    def relations(self):
        return self._relations

    @property
    def facts(self):
        return self._facts

    def get_facts_by_head(self, head: Union[Entity, str]):
        if isinstance(head, str):
            head = Entity(head)
        return self._head_to_facts[head]

    def get_facts_by_relation(self, relation: Union[Relation, str]):
        if isinstance(relation, str):
            relation = Relation(relation)
        return self._relation_to_facts[relation]

    def get_facts_by_tail(self, tail: Union[Entity, str]):
        if isinstance(tail, str):
            tail = Entity(tail)
        return self._tail_to_facts[tail]

    def add(self, fact: Fact):
        self._entities.add(fact.head)
        self._relations.add(fact.relation)
        self._entities.add(fact.tail)
        self._facts.add(fact)

        self._head_to_facts[fact.head].add(fact)
        self._relation_to_facts[fact.relation].add(fact)
        self._tail_to_facts[fact.tail].add(fact)

    def __iadd__(self, other):
        for fact in other.facts:
            self.add(fact)
        return self

    def exact_search(self,
                     head: Optional[Entity],
                     relation: Optional[Relation],
                     tail: Optional[Entity]):
        """Comparing with fuzzy search, exact search is much faster,
        especially when there's lots of facts.
        """
        results: Set[Fact] = set()
        if head:
            results = self.get_facts_by_head(head)

        if relation and not results:
            results = self.get_facts_by_relation(relation)
        elif relation and results:
            results = results.intersection(self.get_facts_by_relation(relation))

        if tail and not results:
            results = self.get_facts_by_tail(tail)
        elif tail and results:
            results = results.intersection(self.get_facts_by_tail(tail))

        return results

    def fuzzy_search(self,
                     head: Optional[Entity],
                     relation: Optional[Relation],
                     tail: Optional[Entity]):
        results: Set[Fact] = set()
        for fact in self.facts:
            if head and not fact.head.is_like(head):
                continue
            if relation and not fact.relation.is_like(relation):
                continue
            if tail and not fact.tail.is_like(tail):
                continue
            results.add(fact)
        return results

    def get_objects(self, category: Union[Entity, str]):
        if isinstance(category, str):
            category = Entity(category)

        objects: Set[Entity] = set()
        for fact in self.get_facts_by_tail(category):
            if fact.relation.is_like(KnowledgeGraph.SUBCATEGORY_RELATION):
                objects.update(self.get_objects(fact.head))
        if not objects:
            return set([category])
        return objects

    @staticmethod
    def load_data(data_dir_path: str):
        return _load_data_recur(data_dir_path, None)


def _load_data_recur(data_dir_path: str,
                     kg: Optional[KnowledgeGraph]):
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
                kg.add(Fact(subcategory,
                            KnowledgeGraph.SUBCATEGORY_RELATION,
                            category))
            update_knowledge_graph(kg, file_path, subcategory)
    return kg


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
        kg.add(Fact(head, KnowledgeGraph.SUBCATEGORY_RELATION, category))
        for relation, tails in relations.items():
            tails = [tails] if isinstance(tails, str) else tails
            for tail in tails:
                kg.add(Fact(head, relation, tail))
