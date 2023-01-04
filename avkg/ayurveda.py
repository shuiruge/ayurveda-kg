import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Set

from knowledge_graph import Entity, KnowledgeGraph, Relation
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

    def select(self, message: str, items: List[str]):
        print(message)
        for i, item in enumerate(items):
            print(f'{i+1}) {item}')
        raw_input = input()
        if not raw_input.isdigit():
            print(f'Error: shall input a number!')
            return self.select(message, items)
        i = int(raw_input)
        if i < 0 or i > len(items):
            print(f'Error: input number shall be between 1 and {len(items)}!')
            return self.select(message, items)
        return items[i-1]

    def rate(self, message: str, items: List[str]):
        print(message)
        scores: List[float] = []
        i = 0
        while i < len(items):
            print(f'{i+1}) {items[i]}:')
            try:
                score = float(input())
                scores.append(score)
                i += 1
            except ValueError:
                print('Shall input a number')
        return scores


class Ayurveda:

    def __init__(self,
                 data_dir_path: str,
                 max_show_symptoms: int = 5):
        self.data_dir_path = data_dir_path
        self.max_show_symptoms = max_show_symptoms

        self._kg = KnowledgeGraph.load_data(data_dir_path)
        self._all_symptoms = self._kg.get_objects('diagnosis')
        self._symptoms: Set[Symptom] = set()
        self._all_doshas = self._kg.get_objects('dosha')
        self._elevated_doshas: Set[Entity] = set()

    def clean_symptoms(self):
        self._symptoms = set()

    def update_symptoms(self):
        with DataCollector() as collector:
            # Select input method.
            msg = 'Select input method:'
            methods = ['direct input', 'input by selection']
            method = collector.select(msg, methods)
            if method == 'direct_input':
                self._update_symptoms_by_input(collector)
            elif method == 'input by selection':
                self._update_symptoms_by_selection(collector)
            else:
                raise ValueError()

    def _update_symptoms_by_input(self, collector: DataCollector):
        raw_symptom = Entity(input())
        for symptom in self._all_symptoms:
            if symptom.is_like(raw_symptom):
                return NotImplemented

    def _update_symptoms_by_selection(self, collector: DataCollector):
        # Select position.
        msg = 'Select the position of your symptom:'
        all_positions = sorted(self._get_all_positions())
        position = collector.select(msg, all_positions)

        # Get all symptoms for that position.
        if position:
            all_symptoms = self._kg.get_objects(position)
        else:
            all_symptoms = self._all_symptoms

        # Display symptoms and get scores.
        msg = 'Rate the symptoms (1~5):'
        show_symptoms = [
            symptom for symptom in all_symptoms
            if symptom not in self._symptoms
        ]
        if len(show_symptoms) > self.max_show_symptoms:
            random.shuffle(show_symptoms)
            num_show = min(len(show_symptoms), self.max_show_symptoms)
            show_symptoms = show_symptoms[:num_show]
        scores = collector.rate(msg, show_symptoms)

        # Update self._symptoms.
        for symptom, score in zip(show_symptoms, scores):
            position = self._get_position(symptom)
            self._symptoms.add(Symptom(str(position), str(symptom), score))

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
        dosha_scores = {dosha: 0 for dosha in self._all_doshas}
        for symptom in self._symptoms:
            facts = self._kg.fuzzy_search(
                head=Entity(symptom.description),
                relation=Relation('hints for elevation'),
            )
            for fact in facts:
                dosha = fact.tail
                dosha_scores[dosha] += symptom.score
        self._elevated_doshas = set(
            get_anomalies(dosha_scores, prob_threshold)
        )

    @property
    def elevated_doshas(self):
        return self._elevated_doshas

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
        pacifies = Relation('pacifies')
        for fact in self._kg.facts:
            if (
                fact.head in all_food and
                fact.relation.is_like(pacifies) and
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

    import os
    data_dir_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data')
    av = Ayurveda(data_dir_path)

    while True:
        print('New start? [y/n]')
        if input() == 'y':
            print("Let's start a new trip!")
            av.clean_symptoms()

        av.update_symptoms()
        print('Symptoms:')
        for symptom in av._symptoms:
            print(symptom)

        av.diagnose()
        print('Elevated doshas:', av.elevated_doshas)
        print('Suggested food:', av.suggest_food())
