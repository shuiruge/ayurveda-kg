from collections import defaultdict
from typing import Dict, Set, Union


class Entity:

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self.name}'

    def is_like(self, name: str):
        # TODO: Temporal implementation without NLP.
        return self.name == name


SUBCATEGORY_RELATION = 'is of'


class Relation:

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
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

    def get_facts_by_head(self, head: Entity):
        if not isinstance(head, Entity):
            head = Entity(head)
        return self._head_to_facts[head]

    def get_facts_by_relation(self, relation: Relation):
        if not isinstance(relation, Relation):
            relation = Relation(relation)
        return self._relation_to_facts[relation]

    def get_facts_by_tail(self, tail: Entity):
        if not isinstance(tail, Entity):
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

    def get_objects(self, category: Entity):
        if not isinstance(category, Entity):
            category = Entity(category)

        objects: Set[Entity] = set()
        for fact in self.get_facts_by_tail(category):
            if fact.relation.is_like(SUBCATEGORY_RELATION):
                objects.update(self.get_objects(fact.head))
        if not objects:
            return set([category])
        return objects

    # ------------- Methods for Ayurveda -------------- #

    def get_food_for_dosha(self, dosha: str):
        all_food = self.get_objects('food')
        food_for_dosha: Set[Entity] = set()
        for fact in self.facts:
            if (
                fact.head in all_food and
                fact.relation.is_like('pacifies') and
                fact.tail.is_like(dosha)
            ):
                food_for_dosha.add(fact.head)
        return food_for_dosha
