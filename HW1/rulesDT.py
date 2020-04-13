import gc
import math
import pandas as pd

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


print(conditional_entropy(rules_df, {0: '1', 1: '0'}))