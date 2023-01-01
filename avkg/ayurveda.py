import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Set

from knowledge_graph import Entity, KnowledgeGraph


class Symptom:

    def __init__(self,
                 description: str,
                 score: float,
                 position: Optional[str] = None):
        self.description = description
        self.score = score
        self.position = position


class DataCollector:

    def select(self, items: List[str]) -> str:
        # TODO: Implement this.
        return NotImplemented

    def rate(self, items: List[str]) -> List[float]:
        # TODO: Implement this.
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
        all_positions = self._get_all_positions()
        with DataCollector() as collector:
            position = collector.select(all_positions)
            if position:
                all_symptoms = self._kg.get_objects(position)
            else:
                all_symptoms = self._kg.get_objects('diagnosis')
            # TODO: Parameterize the 10.
            num_show = min(len(all_symptoms), 10)
            show_symptoms = random.choices(all_symptoms, num_show)
            scores = collector.rate(show_symptoms)
        for symptom, score in zip(show_symptoms, scores):
            self._symptoms.add(Symptom(symptom, score, position))

    def _get_all_positions(self):
        facts = self._kg.exact_search(
            relation=KnowledgeGraph.SUBCATEGORY_RELATION,
            tail='diagnosis',
        )
        all_positions: List[str] = []
        for fact in facts:
            position = str(fact.head)
            all_positions.append(position)
        return all_positions

    def diagnose(self) -> None:
        dosha_scores: Dict[Entity, float] = defaultdict(float)
        for symptom in self._symptoms:
            elevated_doshas = self._kg.fuzzy_search(
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
        for fact in self._kg.facts:
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


def get_anomalies(item_scores: Dict[Hashable, float],
                  threshold: float):
    items = list(item_scores)

    log_scores = []
    for item in items:
        score = item_scores[item]
        assert score >= 0
        log_scores.append(math.log(score + 1e-1))

    probs = softmax(log_scores)

    anomalies = {}
    for item, prob in zip(items, probs):
        if prob > threshold:
            anomalies[item] = prob
    return anomalies


def softmax(xs: List[float]):
    x_max = max(xs)
    xs = [x - x_max for x in xs]
    exp_xs = [math.exp(x) for x in xs]
    denom = sum(exp_xs)
    return [exp_x / denom for exp_x in exp_xs]
