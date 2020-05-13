import logging
import numpy as np
import networkx as nx

from HW1 import utils
from HW1.utils import conditional_entropy

NUM_OF_BITS = 64

node_counter = 0
def node_name(tup):
    global node_counter
    node_counter += 1
    return '{node_counter}:(b{i}=={val}) ' \
        .format(i=tup[0], val=tup[1], node_counter=node_counter)


# def node_name(tup):
#     return '(b{i}=={val}) '.format(i=tup[0], val=tup[1])

def add_nodes(group_count, last_best_bit, which_to_check, to_connect, decision_tree, criteria=None, all_rules=None,
              sub_group_rules=None):
    if len(sub_group_rules) <= group_count or sum(which_to_check) == 1:
        logging.info("sub group size is " + str(len(sub_group_rules)) + ", smaller than " + str(group_count))
        logging.info("sub_group_rules: ")
        logging.info(str(sub_group_rules['rule_number'].index))
        return

    which_to_check[last_best_bit] = 0  # [0,0,0,0,1,1,1,1,1....
    prior_knowledge_zero = (last_best_bit, '0')
    prior_knowledge_one = (last_best_bit, '1')
    try:
        best_zero_bit, rules_zero = criteria(*(sub_group_rules, prior_knowledge_zero, which_to_check))
        best_one_bit, rules_one = criteria(*(sub_group_rules, prior_knowledge_one, which_to_check))
    except Exception as e:
        logging.info("sub group size is " + str(len(sub_group_rules)) + ", its greater than " + str(group_count)
                     + " but all bits known.")
        logging.info("sub_group_rules: ")
        logging.info(str(sub_group_rules['rule_number'].index))
        return

    logging.info("base group:" + str(len(sub_group_rules)))
    logging.info("left tree :" + str(len(rules_zero)))
    logging.info("right tree:" + str(len(rules_one)))
    print("===========")
    print("base group:" + str(len(sub_group_rules)))
    print("left tree :" + str(len(rules_zero)))
    print("right tree:" + str(len(rules_one)))

    # we dont need the table after we split it to 2
    # gc.collect()

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
    first_node, _ = criteria(rules_df, None, to_check)
    logging.info("root node " + str(first_node))
    decision_tree.add_node("root")
    add_nodes(min_group_count, first_node, to_check, "root", decision_tree, criteria, rules_df, rules_df)

    return decision_tree

