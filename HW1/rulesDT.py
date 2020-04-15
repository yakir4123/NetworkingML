import gc
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from netaddr import *

NUM_OF_BITS = 64

rules_df = pd.read_csv('rule01.tsv', sep=' ', index_col=False, header=None,
                       names=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'prot'])
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


def get_rules_by_condition(rules_df, conditions):
    for b_i, value in conditions.items():
        rules_df = rules_df.loc[(rules_df['rule'].str[b_i] == value) |
                                (rules_df['rule'].str[b_i] == '*')]
    return rules_df


def conditional_entropy(rules_df, conditions=None):
    if conditions is not None:
        rules_df = get_rules_by_condition(rules_df, conditions)
    P = [0] * 64
    total_rule_power = rules_df['rule_power'].sum()
    for bi in range(0, 64):
        try:
            P[bi] = (rules_df.loc[(rules_df['rule'].str[bi] == '0'), 'rule_power'].sum() +
                     rules_df.loc[(rules_df['rule'].str[bi] == '*'), 'rule_power'].sum() / 2) / total_rule_power
        except ZeroDivisionError:
            P[bi] = 0

    def entropy(p):
        if p == 1 or p == 0:
            return 0
        return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)

    return [entropy(p) for p in P]


gains = conditional_entropy(rules_df, {})
decision_tree = nx.Graph()
first_node = gains.index(max(gains))
decision_tree.add_node(str(first_node))


def ip_view(d):

    return ''.join(list(map(lambda k: d[k] if k in d.keys() else '_', range(64))))


def add_nodes(last_node, knowledge, which_to_check, to_connect, h):

    if which_to_check.count(0) == len(which_to_check):
        return

    which_to_check[last_node] = 0
    prior_knowledge_zero = knowledge.copy()
    prior_knowledge_one = knowledge.copy()
    prior_knowledge_zero[last_node] = '0'
    prior_knowledge_one[last_node] = '1'
    
    gain_zero = conditional_entropy(rules_df, prior_knowledge_zero)
    gain_zero = [a * b for a, b in zip(gain_zero, which_to_check)]
    zero_node = gain_zero.index(max(gain_zero))
    gain_one = conditional_entropy(rules_df, prior_knowledge_one)
    gain_one = [a * b for a, b in zip(gain_one, which_to_check)]
    one_node = gain_one.index(max(gain_one))

    zero_node_str = ip_view(prior_knowledge_zero)
    print(zero_node_str)
    zero_edge = (to_connect, zero_node_str)
    decision_tree.add_edge(*zero_edge)
    one_node_str = ip_view(prior_knowledge_one)
    print(one_node_str)
    one_edge = (to_connect, one_node_str)
    decision_tree.add_edge(*one_edge)

    add_nodes(zero_node, prior_knowledge_zero.copy(), which_to_check.copy(), zero_node_str, h + 1)
    add_nodes(one_node, prior_knowledge_one.copy(), which_to_check.copy(), one_node_str, h + 1)


to_check = [1] * 64
to_check[first_node] = 0
add_nodes(first_node, {}, to_check, str(first_node), 0)
add_nodes(first_node, {}, to_check, str(first_node), 0)


nx.draw(decision_tree, with_labels=True)
plt.show()