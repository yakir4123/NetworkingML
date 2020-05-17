import logging
import pandas as pd
import networkx as nx

from HW1 import utils
from HW1.utils import conditional_entropy
from sklearn.ensemble import RandomForestRegressor

NUM_OF_BITS = 64


node_counter = 0
def node_name(tup):
    global node_counter
    node_counter += 1
    return '{node_counter}:(b{i}=={val}) ' \
        .format(i=tup[0], val=tup[1], node_counter=node_counter)


def add_nodes(group_count, last_best_bit, which_to_check, to_connect, decision_tree, criteria=None, all_rules=None,
              sub_group_rules=None):
    if len(sub_group_rules) <= group_count or sum(which_to_check) == 1:
        return

    which_to_check[last_best_bit] = 0  # [0,0,0,0,1,1,1,1,1....
    prior_knowledge_zero = (last_best_bit, '0')
    prior_knowledge_one = (last_best_bit, '1')
    try:
        best_zero_bit, rules_zero, _ = criteria(*(sub_group_rules, prior_knowledge_zero, which_to_check))
        best_one_bit, rules_one, _ = criteria(*(sub_group_rules, prior_knowledge_one, which_to_check))
    except Exception as e:
        logging.info("sub group size is " + str(len(sub_group_rules)) + ", its greater than " + str(group_count)
                     + " but all bits are known.")
        logging.info("sub_group_rules: ")
        logging.info(str(sub_group_rules['rule_number'].index))
        return

    logging.info("base group:" + str(len(sub_group_rules)))
    logging.info("left tree :" + str(len(rules_zero)))
    logging.info("right tree:" + str(len(rules_one)))

    zero_node_str = node_name(prior_knowledge_zero)
    logging.info("zero node name: " + zero_node_str)
    zero_edge = (to_connect, zero_node_str)
    decision_tree.add_edge(*zero_edge, object="{ " + str(last_best_bit) + " : 0")

    one_node_str = node_name(prior_knowledge_one)
    logging.info("one node name: " + one_node_str)
    one_edge = (to_connect, one_node_str)
    decision_tree.add_edge(*one_edge, object="{ " + str(last_best_bit) + " : 1")
    logging.info("add childs")
    add_nodes(group_count, best_zero_bit, which_to_check.copy(), zero_node_str, decision_tree, criteria, all_rules,
              rules_zero)
    add_nodes(group_count, best_one_bit, which_to_check.copy(), one_node_str, decision_tree, criteria, all_rules,
              rules_one)


def create_decision_tree(rules_df, min_group_count, criteria):
    gains, _ = conditional_entropy(rules_df, include_wc=(criteria is utils.best_bit_by_IG))
    logging.info("first bit gains: " + str(gains))
    decision_tree = nx.DiGraph()
    to_check = [1] * 64
    first_node, _, _ = criteria(rules_df, None, to_check)
    logging.info("root node " + str(first_node))
    decision_tree.add_node("root")
    add_nodes(min_group_count, first_node, to_check, "root", decision_tree, criteria, rules_df, rules_df)

    return decision_tree


def create_random_forest(rules):
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(rules, pd.DataFrame())
    return forest_model


def create_bfs_tree(rules_df, min_sub_group, criteria):
    logging.info("create bfs tree, min_sub_group = {}".format(min_sub_group))

    decision_tree = nx.DiGraph()
    best_in_level = max if (criteria == utils.best_bit_by_entropy) else min
    logging.info("criteria = {}".format("entropy" if (criteria == utils.best_bit_by_entropy) else "IG"))

    decision_nodes_path = []
    which_to_check = [1] * 64

    best_bit, _, gains = criteria(rules_df, None, which_to_check)
    decision_nodes_path += [best_bit]
    which_to_check[best_bit] = 0
    logging.info("level {}, chosen bit = {}".format(which_to_check.count(0), best_bit))

    decision_tree.add_node("root")
    queue = [("root", rules_df)]
    next_level_queue = []
    next_level_bits_options = []

    while queue:

        # adding new nodes by asking best bit from previous level
        for (rules_name, rules) in queue:
            condition_zero = (best_bit, '0')
            condition_one = (best_bit, '1')
            try:
                _, rules_zero, _ = criteria(rules, condition_zero, which_to_check)
                _, rules_one, _ = criteria(rules, condition_one, which_to_check)
            except Exception:
                logging.info("{} node has {len} > {mgc}, but no entropy left. so I decided to stop"
                             .format(rules_name, len=len(rules), mgc=min_sub_group))
                continue
            zero_name = node_name(condition_zero)
            one_name = node_name(condition_one)
            decision_tree.add_edge(rules_name, zero_name)
            decision_tree.add_edge(rules_name, one_name)
            logging.info("add new node " + zero_name)
            logging.info("add new node " + one_name)

            if len(rules_zero) > min_sub_group:
                next_level_queue += [(zero_name, rules_zero)]
                logging.info("subgroup of " + zero_name + " with size of " + str(len(rules_zero)) + ", will expand it next level")
            else:
                logging.info("subgroup of " + zero_name + " with size of " + str(len(rules_zero)) + ", stop expand it")

            if len(rules_one) > min_sub_group:
                next_level_queue += [(one_name, rules_one)]
                logging.info("subgroup of " + one_name + " with size of " + str(len(rules_one)) + ", will expand it next level")
            else:
                logging.info("subgroup of " + one_name + " with size of " + str(len(rules_one)) + ", stop expand it")

        logging.info("search for next bit.")
        for (rules_name, rules) in next_level_queue:
            best_bit, _, max_gain = criteria(rules, None, which_to_check)
            next_level_bits_options += [(best_bit, max_gain)]
            logging.info("{} candidate: {} bit with gain of {}.".format(rules_name, best_bit, max_gain))

        # find the best bit using best_in_level function
        try:
            best_bit = best_in_level(next_level_bits_options, key=lambda item: item[1])[0]
        except ValueError as e:
            logging.info("finish building the tree, the questions list are:")
            logging.info(str(decision_nodes_path))
            return decision_tree

        next_level_bits_options = []
        decision_nodes_path += [best_bit]
        which_to_check[best_bit] = 0

        if sum(which_to_check) == 0:
            logging.info("Asks about all bits:  {}".format(str(decision_nodes_path)))
            return decision_tree
        logging.info("level {}, chosen bit = {}".format(which_to_check.count(0), best_bit))

        queue = next_level_queue
        next_level_queue = []

