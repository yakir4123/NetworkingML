import gc
import math
import numpy as np
import pandas as pd

import ipaddress

NUM_OF_BITS = 64


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
    ip_addr = ipaddress.ip_network(ip_addr)
    ip2bin = f'{int(ip_addr.network_address):032b}'
    res = ip2bin[:ip_addr.prefixlen] + ((32 - ip_addr.prefixlen) * "*")
    return res


def rule_power(rule):
    return 2 ** rule.count('*')


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


def get_rules_by_condition(rules, condition):
    bit_col = 'b_{}'.format(condition[0])
    rules = rules.loc[(rules[bit_col] == condition[1]) | (rules[bit_col] == '*')].copy()
    return rules


def set_rule_wildcards_bit(rules, condition):
    bit_col = 'b_{}'.format(condition[0])
    rules.loc[(rules[bit_col] == '*'), 'rule_power'] //= 2
    rules.loc[(rules[bit_col] == '*'), bit_col] = condition[1]
    return rules


def best_bit_by_IG(sub_group_rules, given_bit, which_to_check, *args):
    entropy, rules = conditional_entropy(sub_group_rules, given_bit)
    entropy = [2 if b == 0 else a for a, b in
               zip(entropy, which_to_check)]  # [0,0,1,1...  entropy_zero = [0, 0, 0.7...
    best_bit = entropy.index(min(entropy))
    return best_bit, rules


def best_bit_by_entropy(sub_group_rules, prior_knowledge_zero, *args):
    entropy, rules = conditional_entropy(sub_group_rules,
                                         prior_knowledge_zero)  # [0,0,1,1...  entropy_zero = [0, 0, 0.7...
    best_bit = entropy.index(max(entropy))
    return best_bit, rules


def create_packet_table(num_packets):
    packet_df = pd.read_csv('ScrambledPackets01.tsv', sep='\t', index_col=False, header=None,
                            names=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'prot', 'rule'])
    packet_df = packet_df.drop(columns=['src_port', 'dst_port', 'prot'])
    packer_df = packet_df.loc[:num_packets]
    packet_df['src_ip_bits'] = packet_df['src_ip'].apply(ip_to_rule)
    packet_df['dst_ip_bits'] = packet_df['dst_ip'].apply(ip_to_rule)
    packet_df['src_dst_ip_bits'] = packet_df['src_ip_bits'] + packet_df['dst_ip_bits']
    for bit in range(0, NUM_OF_BITS):
        col = 'b_{}'.format(bit)
        packet_df[col] = packet_df['src_dst_ip_bits'].str[bit]
        packet_df[col] = packet_df[col].astype(np.uint8)

    del packet_df['src_ip']
    del packet_df['dst_ip']
    del packet_df['src_ip_bits']
    del packet_df['dst_ip_bits']
    del packet_df['src_dst_ip_bits']
    gc.collect()

    return packet_df


def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", df[col].dtype)

            # make variables for Int, max and min
            mx = df[col].max()
            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            IsInt = -0.01 < result < 0.01

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", df[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return df, NAlist