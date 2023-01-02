import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Set

from knowledge_graph import Entity, KnowledgeGraph
from utils import intersect, softmax


class Symptom:

    def __init__(self,
                 position: str,
                 description: str,
                 score: float):
        self.position = position
        self.description = description
        self.score = score

    def __repr__(self):
        return f'''
        symptom:
            position: {self.position}
            description: {self.description}
            score: {self.score}
        '''


class DataCollector:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

    def select(self, items: List[str]) -> Optional[str]:
        # TODO: Implement this.
        return 'eyes'

    def rate(self, items: List[str]) -> List[float]:
        # TODO: Implement this.
        print('rate items', items)
        return [0.2]


class Ayurveda:

    def __init__(self, data_dir_path: str):
        self.data_dir_path = data_dir_path
        self._kg = KnowledgeGraph.load_data(data_dir_path)

        self._symptoms: Set[Symptom] = set()
        self._elevated_doshas: Set[Entity] = set()

    def clean_symptoms(self):
        self._symptoms = set()

    def update_symptoms(self):
        with DataCollector() as collector:
            # Select position.
            all_positions = self._get_all_positions()
            position = collector.select(all_positions)

            # Get all symptoms for that position.
            if position:
                all_symptoms = self._kg.get_objects(position)
            else:
                all_symptoms = self._kg.get_objects('diagnosis')

            # Display symptoms and get scores.
            # TODO: Parameterize the 10.
            show_symptoms = [
                symptom for symptom in all_symptoms
                if symptom not in self._symptoms
            ]
            if len(show_symptoms) > 10:
                random.shuffle(show_symptoms)
                num_show = min(len(show_symptoms), 10)
                show_symptoms = show_symptoms[:num_show]
            scores = collector.rate(show_symptoms)

        # Update self._symptoms.
        for symptom, score in zip(show_symptoms, scores):
            position = self._get_position(symptom)
            self._symptoms.add(Symptom(position, symptom, score))

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

    def _get_position(self, symptom: str):
        facts = self._kg.exact_search(
            head=symptom,
            relation=KnowledgeGraph.SUBCATEGORY_RELATION,
        )
        for fact in facts:
            return str(fact.tail)
        raise ValueError(f'Cannot find position for symptom {symptom}')

    def diagnose(self, prob_threshold: float = 0.5) -> None:
        dosha_scores: Dict[Entity, float] = defaultdict(float)
        for symptom in self._symptoms:
            facts = self._kg.fuzzy_search(
                head=symptom.description,
                relation='hints for elevation',
            )
            for fact in facts:
                dosha = fact.tail
                dosha_scores[dosha] += symptom.score
        self._elevated_doshas = set(
            get_anomalies(dosha_scores, prob_threshold)
        )

    def suggest_food(self) -> Set[Entity]:
        if not self._elevated_doshas:
            return set()

        food_for_doshas = []
        for dosha in self._elevated_doshas:
            food = self._suggest_food_for_dosha(dosha)
            if food:
                food_for_doshas.append(food)
        return intersect(*food_for_doshas)

    def _suggest_food_for_dosha(self, dosha: Entity):
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


def get_anomalies(item_scores: Dict[Hashable, float],
                  prob_threshold: float):
    items = list(item_scores)

    log_scores = []
    for item in items:
        score = item_scores[item]
        assert score >= 0
        log_scores.append(math.log(score + 1e-1))

    probs = softmax(log_scores)

    anomalies = {}
    for item, prob in zip(items, probs):
        if prob > prob_threshold:
            anomalies[item] = prob
    return anomalies


if __name__ == '__main__':

    av = Ayurveda('../data')
    av.update_symptoms()
    print('Symptoms:', av._symptoms)
    av.diagnose()
    print('Elevated doshas:', av._elevated_doshas)
    print('Suggested food:', av.suggest_food())
