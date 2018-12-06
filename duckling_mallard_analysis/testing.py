import requests
import json
import re
import pickle
import pprint
from collections import defaultdict, Counter


MALLARD_ENDPOINT = "http://localhost:2626/parse"
LANGUAGE = 'eng'

DUCKLING_ENDPOINT = "http://0.0.0.0:8000/parse"


def remove_role_and_sys(x):
    if '|' in x:
        index = x.index('|')
        return x[4:index]
    else:
        return x[4:]


def get_expected_spans(clean, labeled):
    """
    Extract system entities from each query, with start, end, and label for each entity

    :param clean: queries with no entity labels
    :param labeled: queries with entity labels
    :return: entity spans for each query, each entity span in form (start_index, end_index, label)
    """
    all_spans = []
    for clean_query, labeled_query in zip(clean, labeled):
        # Actual entity spans in the clean query
        spans = []
        entities = re.findall('{([^{]+)\|sys_[^\s]*}', labeled_query)
        entity_labels = re.findall('{[^{]+\|(sys_[^\s]*)}', labeled_query)
        entity_labels = [remove_role_and_sys(x) for x in entity_labels]

        # Advance through the query so we don't get repeat spans
        start_position = 0
        for e, label in zip(entities, entity_labels):
            start_index = clean_query.index(e, start_position)
            end_index = start_index + len(e)
            spans.append((start_index, end_index, label))

            start_position = end_index

        all_spans.append(spans)

    return all_spans


def get_mallard_results(queries):
    """
    Return results of each Mallard call on our queries
    :param queries: Clean queries with no labels
    :return:
    """
    responses = []

    for i, query in enumerate(queries):

        if i % 1000 == 0:
            print(i)

        data = {
            'text': query,
            'language': LANGUAGE
        }

        try:
            response = requests.request('POST', MALLARD_ENDPOINT, json=data)
            response = response.json()
            responses.append(response['data'])
        except Exception as ex:
            print('Numerical Entity Recognizer Error %s\nURL: %r\nData: %s',
                  ex, MALLARD_ENDPOINT, json.dumps(data))

    return responses


def get_duckling_results(queries):
    """
    Return results of each Duckling call on our queries
    :param queries: Clean queries with no labels
    :return:
    """
    responses = []

    for i, query in enumerate(queries):

        if i % 1000 == 0:
            print(i)

        data = {
            'text': query,
        }

        try:
            response = requests.request('POST', DUCKLING_ENDPOINT, data=data)
            response = response.json()
            responses.append(response)
        except Exception as ex:
            print('Numerical Entity Recognizer Error %s\nURL: %r\nData: %s',
                  ex, MALLARD_ENDPOINT, json.dumps(data))

    return responses


def parse_mallard_response(response):
    """
    Gets all predicted entities from a single Mallard response
    :param response: Mallard response
    :return: dict with keys being dimensions and values being another dict that maps span to value
    """
    possible_entities = defaultdict(dict)

    try:
        for r in response:
            dimension = r['dimension']
            span = (r['entity']['start'], r['entity']['end'])

            value = r['value'][0]

            possible_entities[dimension][span] = value
    except Exception as exp:
        print(dimension)
        print(exp)
        print(r)

    return possible_entities


def parse_duckling_response(response):
    """
    Gets all predicted entities from a single Duckling response
    :param response: Duckling response
    :return: dict with keys being dimensions and values being another dict that maps span to value
    """
    possible_entities = defaultdict(dict)

    for r in response:
        dimension = r['dim']
        span = (r['start'], r['end'])
        value = None

        if dimension == 'time':
            type_ = r['value']['type']
            # Single time
            if type_ == 'value':
                value = r['value']['value']
            # Time intervals
            elif type_ == 'interval':
                value = (r['value'].get('from', None), r['value'].get('to', None))
            else:
                print("UNEXPECTED TIME VALUE")

        elif dimension == 'temperature':
            type_ = r['value']['type']
            if type_ == 'value':
                value = r['value']['value']
            else:
                value = (r['value'].get('from', None), r['value'].get('to', None))

        elif dimension == 'amount-of-money':
            type_ = r['value']['type']
            if type_ == 'value':
                value = r['value']['value']
            else:
                value = (r['value'].get('from', None), r['value'].get('to', None))
        else:
            value = r['value']['value']

        possible_entities[dimension][span] = value

    return possible_entities


def evaluate_ser(entity, outputs):
    """
    Get all correct and incorrect queries. Correct means all entities are identified correctly
    with exact span

    :param entity: list of (start_index, end_index, label) tuples for EACH query
    :param outputs: PARSED mallard/duckling responses
    :return: Two lists of indicies, one for correct queries and one for incorrect
    """
    incorrect_queries = []
    correct_queries = []

    for i, (entity_info, output) in enumerate(zip(entity, outputs)):
        correct = True

        # For each entity, see if it is present in the mallard output
        for entity in entity_info:
            span = (entity[0], entity[1])
            entity_type = entity[2]

            if entity_type not in output:
                correct = False
            else:
                if span not in output[entity_type]:
                    correct = False
        if correct:
            correct_queries.append(i)
        else:
            incorrect_queries.append(i)

    return correct_queries, incorrect_queries


def evaluate_ser_errors(entities, outputs):
    """
    # For queries that the ser gets incorrect, group them into different types
    :param entities: In the form [(5, 6, 'number')], list of entities for each query
    :param outputs: PARSED mallard/duckling responses
    :return:
    """
    missed_entity_indices = []
    incorrect_span_indices = []
    correct_indices = []

    for i, (entity_info, output) in enumerate(zip(entities, outputs)):

        missed_entity = False
        incorrect_span = False

        for entity in entity_info:
            span = (entity[0], entity[1])
            entity_type = entity[2]

            if entity_type not in output:
                # Completely not predicted
                if i not in missed_entity_indices:
                    missed_entity_indices.append(i)
                missed_entity = True
            else:
                if span not in output[entity_type]:
                    if i not in incorrect_span_indices:
                        incorrect_span_indices.append(i)
                    incorrect_span = True

        if not missed_entity and not incorrect_span:
            correct_indices.append(i)

    return missed_entity_indices, incorrect_span_indices, correct_indices


def compare_mallard_duckling(mallards, ducklings):
    """
    For each query, go through each duckling prediction to see if an IDENTICAL prediction was
    made by Mallard as well
    :param mallards: parsed Mallard output
    :param ducklings: parsed Duckling output
    :return:
    """
    present = []
    not_present = []

    for i, (mallard, duckling) in enumerate(zip(mallards, ducklings)):
        same = True

        for type_, infos in duckling.items():
            if type_ == 'amount-of-money':
                continue
            if type_ not in mallard:
                same = False
            else:
                duckling_spans = list(infos.keys())
                mallard_spans = list(mallard[type_].keys())

                for d in duckling_spans:
                    if d not in mallard_spans:
                        same = False
        if same:
            present.append(i)
        else:
            not_present.append(i)

    return present, not_present


def find_duckling_conflict_queries(duckling_outputs):
    """
    Finds queries where duckling predicts multiple entities for the SAME span
    :param duckling_outputs: PARSED duckling responses, dicts from dimension to a dict mapping tuple spans to values
    :return:
    """
    conflict_responses = {}

    for i, response in enumerate(duckling_outputs):
        response_values = list(response.values())
        response_spans = []

        for rv in response_values:
            spans = list(rv.keys())
            response_spans.extend(spans)

        if len(response_spans) != len(set(response_spans)):
            conflict_responses[i] = response

    return conflict_responses


if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=2)

    with open("sys_queries.txt", "r") as f:
        queries_labeled = f.readlines()
        queries_labeled = [x.strip() for x in queries_labeled]

    with open("sys_queries_clean.txt", "r") as f:
        queries_clean = f.readlines()
        queries_clean = [x.strip() for x in queries_clean]

    entity_spans = get_expected_spans(queries_clean, queries_labeled)

    # Test that the Mallard return matches each retrieved span
    # duckling_results = get_duckling_results(queries_clean)
    # mallard_results = get_mallard_results(queries_clean)

    # Load from pickle to save time
    duckling_results = pickle.load(open("duckling_results.p", "rb"))
    mallard_results = pickle.load(open("mallard_results.p", "rb"))

    # Parse the entity spans/values from duckling and mallard
    duckling_outputs = []
    mallard_outputs = []

    for i, dr in enumerate(duckling_results):
        duckling_entities = parse_duckling_response(dr)
        duckling_outputs.append(duckling_entities)

    for mr in mallard_results:
        mallard_entities = parse_mallard_response(mr)
        mallard_outputs.append(mallard_entities)

    # Compare to ground truth labels
    correct_mallard, incorrect_mallard = evaluate_ser(entity_spans, mallard_outputs)  # (6562, 5803)
    correct_duckling, incorrect_duckling = evaluate_ser(entity_spans, duckling_outputs)  # (8169, 4196)

    present, not_present = compare_mallard_duckling(mallard_outputs, duckling_outputs)  # (8764, 3601)

    # Queries that duckling gets wrong but mallard doesn't get wrong
    duckling_regressions = list(set(incorrect_duckling) - set(incorrect_duckling).intersection(set(incorrect_mallard)))

    # Get queries where Duckling predicts different entities for the same span, 996 total
    conflict_responses = find_duckling_conflict_queries(duckling_outputs)


    # Write the different types of errors to file
    # Want to figure out which queries Mallard completely missed out on
    duckling_regressions_queries_labeled = [queries_labeled[i] for i in duckling_regressions]
    duckling_regressions_queries_clean = [queries_clean[i] for i in duckling_regressions]
    duckling_regression_entity_spans = [entity_spans[i] for i in duckling_regressions]
    duckling_regression_outputs = [duckling_outputs[i] for i in duckling_regressions]


    # Remember these indicies correspond to the above duckling things
    missed_entity_indices, incorrect_span_indices, correct_indices = evaluate_ser_errors(
        duckling_regression_entity_spans, duckling_regression_outputs)
    # 564, 71, 0 for duckling regressions

    difference_counts = defaultdict(list)

    with open('duckling_missed_entities.txt', 'w') as f:
        for i in missed_entity_indices:
            # if len(list(duckling_regression_outputs[i].keys())) > 1:
            #     print(i)
            expected_sys_entities = [x[2] for x in duckling_regression_entity_spans[i]]
            actual_sys_entities = list(duckling_regression_outputs[i].keys())

            f.write(f"Index: {i}\n")
            f.write("Query: " + duckling_regressions_queries_labeled[i] + '\n')
            f.write("Expected Sys Entities: " + ", ".join(expected_sys_entities) + '\n')
            f.write("Actual Sys Entities: " + ", ".join(actual_sys_entities) + '\n')
            f.write("\n")

            for k, expected_pred in enumerate(expected_sys_entities):

                if k >= len(actual_sys_entities):
                    actual_pred = 'MISSING'
                else:
                    actual_pred = actual_sys_entities[k]

                if expected_pred != actual_pred:
                    difference_counts[expected_pred].append(actual_pred)

    for dim, mistakes in difference_counts.items():
        difference_counts[dim] = Counter(mistakes)

    difference_counts = dict(difference_counts)

    difference_count_totals = {}
    for dim, counts in difference_counts.items():
        difference_count_totals[dim] = sum(counts.values())
