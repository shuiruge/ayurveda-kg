import random
from collections import defaultdict
from typing import Dict, Hashable, List, Set

from knowledge_graph import Entity, KnowledgeGraph


class Symptom:

    def __init__(self, position: str, description: str, score: float):
        self.position = position
        self.description = description
        self.score = score


class DataCollector:

    def input(self) -> str:
        return NotImplemented

    def interact(self, display: List[str]) -> List[float]:
        return NotImplemented


class Ayurveda:

    def __init__(self, data_dir_path: str):
        self.data_dir_path = data_dir_path
        self._kg = KnowledgeGraph.load_data(data_dir_path)

        self._symptoms: Set[Symptom] = set()
        self._elevated_doshas: Set[Entity] = set()

    def clean_symptoms(self):
        self._symptoms = set()

    def update_symptoms(self):
        possible_symptoms = self._get_possible_symptoms(position)
        with DataCollector() as collector:
            position = collector.input()
            if not position:
                all_symptoms = self._kg.get_objects('diagnosis')
            elif not self._kg.exact_search(
                    head=position,
                    relation=KnowledgeGraph.SUBCATEGORY_RELATION,
                    tail='diagnosis',
                ):
                    raise ValueError(
                        f'Position "{position}" is not in the diagnosis list.')
            else:
                all_symptoms = self._kg.get_objects(position)
            # TODO: Parameterize the 10.
            num_show = min(len(all_symptoms), 10)
            show_symptoms = random.choices(all_symptoms, num_show)
            scores = collector.display(show_symptoms)
        for symptom, score in zip(show_symptoms, scores):
            self._symptoms.add(Symptom(position, symptom, score))

    def diagnose(self) -> None:
        dosha_scores: Dict[Entity, float] = defaultdict(float)
        for symptom in self._symptoms:
            elevated_doshas = self._kg.exact_search(
                head=symptom.description,
                relation='hints for elevation',
            )
            for dosha in elevated_doshas:
                dosha_scores[dosha] += symptom.score
        self._elevated_doshas = set(get_anomalies(dosha_scores))

    def suggest_food(self) -> Set[Entity]:
        if not self._elevated_doshas:
            return set()

        food_for_doshas = [
            self._get_food_for_dosha(dosha)
            for dosha in self._elevated_doshas
        ]
        return intersect(*food_for_doshas)

    def _get_food_for_dosha(self, dosha: Entity):
        all_food = self._kg.get_objects('food')
        food_for_dosha: Set[Entity] = set()
        for fact in self.facts:
            if (
                fact.head in all_food and
                fact.relation.is_like('pacifies') and
                fact.tail.is_like(dosha)
            ):
                food_for_dosha.add(fact.head)
        return food_for_dosha


def intersect(*args: set):
    if not args:
        return set()
    arg0, *rest_args = args
    return arg0.intersection(intersect(*rest_args))


def get_anomalies(scores: Dict[Hashable, float],
                  threshold: float) -> List[Hashable]:
    return NotImplemented
