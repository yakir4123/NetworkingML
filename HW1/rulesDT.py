import gc
import math
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from netaddr import *

from HW1 import utils

NUM_OF_BITS = 64
GROUP_NUM = 128

rules_df = pd.read_csv('rule01.tsv', sep=' ', index_col=False, header=None,
                       names=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'prot'])
rules_df = rules_df.loc[:400]
rules_df = rules_df.drop(columns=['src_port', 'dst_port', 'prot'])
rules_df['src_ip'] = rules_df['src_ip'].apply(lambda s: s[1:])


def ip_to_rule(ip_addr):
    ip_presentation = IPNetwork(ip_addr)
    net = ip_presentation.network.bin[2:]
    net = (32 - len(net)) * "0" + net
    net = net[:ip_presentation.prefixlen] + ((32 - ip_presentation.prefixlen) * "*")
    return net


def rule_power(rule):
    return 2 ** rule.count('*')


# Yakir - process the data
rules_df['src_ip_rule'] = rules_df['src_ip'].apply(ip_to_rule)
rules_df['dst_ip_rule'] = rules_df['dst_ip'].apply(ip_to_rule)
rules_df['rule'] = rules_df['src_ip_rule'] + rules_df['dst_ip_rule']
rules_df['rule_power'] = rules_df['rule'].apply(rule_power)

del rules_df['src_ip_rule']
del rules_df['dst_ip_rule']
gc.collect()


def get_rules_by_condition(rules, condition):
    rules = rules.loc[(rules['rule'].str[condition[0]] == condition[1]) |
                      (rules['rule'].str[condition[0]] == '*')]
    return rules


def set_rule_wildcards_bit(rules, condition):
    rules.loc[(rules['rule'].str[condition[0]] == '*'), 'rule_power'] //= 2
    rules.loc[(rules['rule'].str[condition[0]] == '*'), 'rule'] =\
        rules.loc[(rules['rule'].str[condition[0]] == '*'), 'rule'].str[:condition[0]]\
        + condition[1]\
        + rules.loc[(rules['rule'].str[condition[0]] == '*'), 'rule'].str[condition[0]+1:]
    return rules


def conditional_entropy(rules, conditions=None):
    if conditions is not None:
        rules = get_rules_by_condition(rules, conditions)
    P = [0] * 64
    total_rule_power = rules['rule_power'].sum()
    for bi in range(0, 64):
        try:
            P[bi] = (rules.loc[(rules['rule'].str[bi] == '0'), 'rule_power'].sum() +
                     rules.loc[(rules['rule'].str[bi] == '*'), 'rule_power'].sum() / 2) / total_rule_power
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


def add_nodes(last_node, which_to_check, to_connect, rules):
    if len(rules) <= GROUP_NUM:
        return

    which_to_check[last_node] = 0  # [0,0,0,0,1,1,1,1,1....
    prior_knowledge_zero = (last_node, '0')
    prior_knowledge_one = (last_node, '1')

    gain_zero, rules_zero = conditional_entropy(rules, prior_knowledge_zero)
    gain_zero = [-a * b for a, b in zip(gain_zero, which_to_check)]  # [0,0,1,1...  gain_zero = [0, 0, 0.7...
    zero_node = gain_zero.index(min(gain_zero))
    gain_one, rules_one = conditional_entropy(rules, prior_knowledge_one)
    gain_one = [-a * b for a, b in zip(gain_one, which_to_check)]
    one_node = gain_one.index(min(gain_one))

    zero_node_str = node_name(prior_knowledge_zero)
    print(zero_node_str)
    zero_edge = (to_connect, zero_node_str)
    decision_tree.add_edge(*zero_edge, object="{ " + str(last_node) + " : 0")
    one_node_str = node_name(prior_knowledge_one)
    print(one_node_str)
    one_edge = (to_connect, one_node_str)
    decision_tree.add_edge(*one_edge, object="{ " + str(last_node) + " : 1")

    add_nodes(zero_node, which_to_check.copy(), zero_node_str, rules_zero)
    add_nodes(one_node, which_to_check.copy(), one_node_str, rules_one)


start_time = time.time()
gains, _ = conditional_entropy(rules_df)
decision_tree = nx.Graph()
first_node = gains.index(min(gains))
decision_tree.add_node(str(first_node))

to_check = [1] * 64
add_nodes(first_node, to_check, str(first_node), rules_df)

pos = utils.hierarchy_pos(decision_tree, 1)
nx.draw(decision_tree, pos, with_labels=True)

nx.write_adjlist(decision_tree, "decision_tree_{}".format(GROUP_NUM))
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
