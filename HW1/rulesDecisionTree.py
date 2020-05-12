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

    if criteria is not None:
        best_zero_bit, rules_zero = criteria(*(sub_group_rules, prior_knowledge_zero, which_to_check))
        best_one_bit, rules_one = criteria(*(sub_group_rules, prior_knowledge_one, which_to_check))
    else:
        entropy, _ = np.array(conditional_entropy(all_rules))
        level = which_to_check.count(0)
        sorted_entropy = entropy.copy()
        sorted_entropy.sort()
        entropy = [2 if b == 0 else a for a, b in
                   zip(entropy, which_to_check)]  # [0,0,1,1...  entropy_zero = [0, 0, 0.7...
        best_bit = entropy.index(sorted_entropy[level])

        _, rules_zero = conditional_entropy(sub_group_rules, prior_knowledge_zero)
        _, rules_one = conditional_entropy(sub_group_rules, prior_knowledge_one)
        # to have one code for both cases simply "duplicate" the values for both trees
        best_zero_bit = best_bit
        best_one_bit = best_bit
    print("===========")
    print("base group:" + str(len(sub_group_rules)))
    print("left tree :" + str(len(rules_zero)))
    print("right tree:" + str(len(rules_one)))

    # we dont need the table after we split it to 2
    # gc.collect()

    zero_node_str = node_name(prior_knowledge_zero)
    zero_edge = (to_connect, zero_node_str)
    decision_tree.add_edge(*zero_edge, object="{ " + str(last_best_bit) + " : 0")
    one_node_str = node_name(prior_knowledge_one)
    one_edge = (to_connect, one_node_str)
    decision_tree.add_edge(*one_edge, object="{ " + str(last_best_bit) + " : 1")

    add_nodes(group_count, best_zero_bit, which_to_check.copy(), zero_node_str, decision_tree, criteria, all_rules,
              rules_zero)
    add_nodes(group_count, best_one_bit, which_to_check.copy(), one_node_str, decision_tree, criteria, all_rules,
              rules_one)


def create_decision_tree(rules_df, min_group_count, criteria):
    gains, _ = conditional_entropy(rules_df)
    decision_tree = nx.DiGraph()
    to_check = [1] * 64
    if criteria is not None:
        first_node, _ = criteria(rules_df, None, to_check)
    else:
        first_node, _ = utils.best_bit_by_IG(rules_df, None, to_check)
    decision_tree.add_node("root")
    add_nodes(min_group_count, first_node, to_check, "root", decision_tree, criteria, rules_df, rules_df)

    return decision_tree


def create_random_forest(rules):
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(rules, pd.DataFrame())
    return forest_model

