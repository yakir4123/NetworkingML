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


def create_tree(rules_df, min_group_count, criteria, is_max):

    gains, rules = conditional_entropy(rules_df)
    decision_tree = nx.Graph()
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
    which_to_check[first_node] = 0
    queue = [rules]
    decision_tree.add_node(str(best_bit))
    print("first bit: " + str(best_bit))
    print("rule_len: " + str(len(rules)))
    to_connect = [str(best_bit)]
    which_to_connect = []

    while queue != [] and 1 < sum(which_to_check) < 126:

        gains_compare = []
        best_index = []
        next_queue = []
        condition_zero = (best_bit, '0')
        condition_one = (best_bit, '1')
        temp = []

        while queue != []:

            if len(queue[0]) > min_group_count:
                temp += [to_connect.pop(0)]
                pop = queue.pop(0)
                _, rules_zero, max_gain_zero = criteria(pop, condition_zero)
                _, rules_one, max_gain_one = criteria(pop, condition_one)
                print('len of rules_zero :' + str(len(rules_zero)))
                print('len of rules_one :' + str(len(rules_one)))

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
        print("best bit: " + str(best_bit))
        which_to_check[best_bit] = 0
        queue = next_queue

        print("num to connect: " + str(len(to_connect)))

        temp = []
        for index in which_to_connect:

            node_zero = node_name((best_bit, '0'))
            node_one = node_name((best_bit, '1'))
            print("nodes: ",node_zero,node_one)

            decision_tree.add_edge(to_connect[0], node_zero, object="{ " + str(prev) + " : 0")
            decision_tree.add_edge(to_connect.pop(0), node_one, object="{ " + str(prev) + " : 1")
            temp += [node_zero, node_one]

        which_to_connect = []
        to_connect = temp

    print(str(decision_tree._adj))

    return decision_tree, decision_nodes_path
