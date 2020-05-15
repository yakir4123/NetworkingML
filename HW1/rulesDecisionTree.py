import logging
import numpy as np
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


# def node_name(tup):
#     return '(b{i}=={val}) '.format(i=tup[0], val=tup[1])

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
    first_node, _, _ = criteria(rules_df, None, to_check)
    logging.info("root node " + str(first_node))
    decision_tree.add_node("root")
    add_nodes(min_group_count, first_node, to_check, "root", decision_tree, criteria, rules_df, rules_df)

    return decision_tree


def create_random_forest(rules):
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(rules, pd.DataFrame())
    return forest_model


def create_tree(rules_df, min_group_count, criteria, is_max):
    gains, rules = conditional_entropy(rules_df)
    decision_tree = nx.Graph()
    decision_tree = decision_tree.to_directed()
    decision_nodes_path = []
    i = 0
    if is_max:
        first_node = gains.index(max(gains))
        which_to_check = [1] * 64
    else:
        first_node = gains.index(min(gains))
        which_to_check = [2] * 64

    best_bit = first_node
    decision_nodes_path += [best_bit]

    decision_tree.add_node("root")

    print("root: {best}".format(best=best_bit))
    which_to_check[first_node] = 0
    queue = [rules]
    to_connect = ["root"]
    which_to_connect = []

    while queue != [] and 1 < sum(which_to_check) < min_group_count:

        gains_compare = []
        best_index = []
        next_queue = []
        condition_zero = (best_bit, '0')
        condition_one = (best_bit, '1')
        temp = []

        while queue:

            if len(queue[0]) > min_group_count:
                print(to_connect[0], ":")
                temp += [to_connect.pop(0)]
                pop = queue.pop(0)
                _, rules_zero, max_gain_zero = criteria(pop, condition_zero)
                _, rules_one, max_gain_one = criteria(pop, condition_one)

                print('left node {i}:(b{last}==0), num of rules: {num}'
                      .format(i=i, last=decision_nodes_path[-1], num=str(len(rules_zero))))
                print('right node {i}:(b{last}==1), num of rules: {num}'
                      .format(i=i+1, last=decision_nodes_path[-1], num=str(len(rules_one))))

                gains_to_check_zero = [max_gain_zero[i] * which_to_check[i] for i in range(64)]
                gains_to_check_one = [max_gain_one[i] * which_to_check[i] for i in range(64)]
                if is_max:
                    best_zero = max(gains_to_check_zero)
                    best_one = max(gains_to_check_one)
                else:
                    best_zero = min(gains_to_check_zero)
                    best_one = min(gains_to_check_one)
                zero_index = gains_to_check_zero.index(best_zero)
                one_index = gains_to_check_one.index(best_one)

                next_queue += [rules_zero, rules_one]
                gains_compare += [best_zero, best_one]
                best_index += [zero_index, one_index]
                which_to_connect += [i]

            else:
                queue.pop(0)
                to_connect.pop(0)

            i += 2

        if len(gains_compare) == 0 or len(gains_compare) == 128:
            break

        prev = best_bit
        to_connect = temp
        if is_max:
            best_gain_index = gains_compare.index(max(gains_compare))
        else:
            best_gain_index = gains_compare.index(min(gains_compare))

        best_bit = best_index[best_gain_index]
        decision_nodes_path += [best_bit]
        which_to_check[best_bit] = 0
        queue = next_queue

        temp = []
        for index in which_to_connect:

            node_zero = node_name((best_bit, '0'))
            node_one = node_name((best_bit, '1'))

            decision_tree.add_edge(to_connect[0], node_zero, object="{ " + str(prev) + " : 0")
            decision_tree.add_edge(to_connect.pop(0), node_one, object="{ " + str(prev) + " : 1")
            temp += [node_zero, node_one]

        which_to_connect = []
        to_connect = temp

    print(decision_nodes_path)

    return decision_tree, decision_nodes_path
