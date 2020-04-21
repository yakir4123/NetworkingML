import gc
import math
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from netaddr import *
from HW1 import utils

NUM_OF_BITS = 64
GROUP_NUM = 128
IS_BEST_BIT = True


def create_rule_table():
    rules_df = pd.read_csv('rule01.tsv', sep=' ', index_col=False, header=None,
                           names=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'prot'])
    rules_df = rules_df.drop(columns=['src_port', 'dst_port', 'prot'])
    rules_df['src_ip'] = rules_df['src_ip'].apply(lambda s: s[1:])
    rules_df['src_ip_rule'] = rules_df['src_ip'].apply(ip_to_rule)
    rules_df['dst_ip_rule'] = rules_df['dst_ip'].apply(ip_to_rule)
    rules_df['rule'] = rules_df['src_ip_rule'] + rules_df['dst_ip_rule']
    for bit in range(0, NUM_OF_BITS):
        rules_df['b_{}'.format(bit)] = rules_df['rule'].str[bit]
    rules_df['rule_power'] = rules_df['rule'].apply(rule_power)

    del rules_df['src_ip_rule']
    del rules_df['src_ip']
    del rules_df['dst_ip_rule']
    del rules_df['dst_ip']
    del rules_df['rule']
    gc.collect()

    return rules_df


def ip_to_rule(ip_addr):
    ip_presentation = IPNetwork(ip_addr)
    net = ip_presentation.network.bin[2:]
    net = (32 - len(net)) * "0" + net
    net = net[:ip_presentation.prefixlen] + ((32 - ip_presentation.prefixlen) * "*")
    return net


def rule_power(rule):
    return 2 ** rule.count('*')


def get_rules_by_condition(rules, condition):
    bit_col = 'b_{}'.format(condition[0])
    rules = rules.loc[(rules[bit_col] == condition[1]) | (rules[bit_col] == '*')].copy()
    return rules


def set_rule_wildcards_bit(rules, condition):
    bit_col = 'b_{}'.format(condition[0])
    rules.loc[(rules[bit_col] == '*'), 'rule_power'] //= 2
    rules.loc[(rules[bit_col] == '*'), bit_col] = condition[1]
    return rules


def conditional_entropy(rules, conditions=None):
    if conditions is not None:
        rules = get_rules_by_condition(rules, conditions)
    P = [0] * 64
    total_rule_power = rules['rule_power'].sum()
    for bi in range(0, NUM_OF_BITS):
        try:
            bit_col = 'b_{}'.format(bi)
            P[bi] = (rules.loc[(rules[bit_col] == '0'), 'rule_power'].sum() +
                     rules.loc[(rules[bit_col] == '*'), 'rule_power'].sum() / 2) / total_rule_power
        except ZeroDivisionError:
            P[bi] = 0

    def entropy(p):
        if p == 1 or p == 0:
            return 0
        return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)
    if conditions is not None:
        rules = set_rule_wildcards_bit(rules, conditions)
    return [entropy(p) for p in P], rules


node_counter = 0
def node_name(tup):
    global node_counter
    node_counter += 1
    return '{node_counter}:b_{i}={val}'\
        .format(i=tup[0], val=tup[1], node_counter=node_counter)


def best_bit_by_IG(sub_group_rules, prior_knowledge_zero, which_to_check, *args):
    entropy, rules = conditional_entropy(sub_group_rules, prior_knowledge_zero)
    entropy = [2 if b == 0 else a for a, b in
                    zip(entropy, which_to_check)]  # [0,0,1,1...  entropy_zero = [0, 0, 0.7...
    best_bit = entropy.index(min(entropy))
    return best_bit, rules


def best_bit_by_entropy(sub_group_rules, prior_knowledge_zero, *args):
    entropy, rules = conditional_entropy(sub_group_rules, prior_knowledge_zero)  # [0,0,1,1...  entropy_zero = [0, 0, 0.7...
    best_bit = entropy.index(max(entropy))
    return best_bit, rules


def add_nodes(last_best_bit, which_to_check, to_connect, decision_tree, criteria, all_rules=None, sub_group_rules=None):

    if len(sub_group_rules) <= GROUP_NUM or sum(which_to_check) == 1:
        return

    which_to_check[last_best_bit] = 0  # [0,0,0,0,1,1,1,1,1....
    prior_knowledge_zero = (last_best_bit, '0')
    prior_knowledge_one = (last_best_bit, '1')

    if IS_BEST_BIT:
        best_zero_bit, rules_zero = criteria(*(sub_group_rules, prior_knowledge_zero, which_to_check))
        best_one_bit, rules_one = criteria(*(sub_group_rules, prior_knowledge_one, which_to_check))
    else:
        entropy, _ = np.array(conditional_entropy(all_rules))
        level = which_to_check.count(0)
        sorted_entropy = entropy.copy()
        sorted_entropy.sort()
        entropy = [2 if b == 0 else a for a, b in zip(entropy, which_to_check)]  # [0,0,1,1...  entropy_zero = [0, 0, 0.7...
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

    add_nodes(best_zero_bit, which_to_check.copy(), zero_node_str, decision_tree, criteria, all_rules, rules_zero)
    add_nodes(best_one_bit, which_to_check.copy(), one_node_str, decision_tree, criteria, all_rules, rules_one)


def main():
    start_time = time.time()
    rules_df = create_rule_table()
    gains, _ = conditional_entropy(rules_df)
    decision_tree = nx.Graph()
    first_node = gains.index(min(gains))
    decision_tree.add_node(str(first_node))

    to_check = [1] * 64
    add_nodes(first_node, to_check, str(first_node), decision_tree, best_bit_by_entropy, rules_df, rules_df)

    nx.write_adjlist(decision_tree, "decision_tree_{BestBit}BB_{SIZE}".format(SIZE=GROUP_NUM, BestBit="" if IS_BEST_BIT else "Non"))
    
    pos = utils.hierarchy_pos(decision_tree, 1)
    nx.draw(decision_tree, pos, with_labels=True)

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))


main()
