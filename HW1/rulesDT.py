import gc
import math
import numpy as np
import pandas as pd

from netaddr import *
from operator import add, truediv

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


# Yakir - process the data
rules_df['src_ip_rule'] = rules_df['src_ip'].apply(ip_to_rule)
rules_df['dst_ip_rule'] = rules_df['dst_ip'].apply(ip_to_rule)
rules_df['rule'] = rules_df['src_ip_rule'] + rules_df['dst_ip_rule']

del rules_df['src_ip_rule']
del rules_df['dst_ip_rule']
# rules_df = rules_df.drop(columns=['src_ip_rule', 'dst_ip_rule'])

gc.collect()
binary = rules_df['rule'].str.replace('*','2')[:10]


def best_gain(given_bits):

    res = binary
    for key in given_bits.keys():
        res = list(filter(lambda i: i[key] == given_bits[key], res))
    print(len(res))

    def is_two(bit):
        if bit == '2':
            return 1
        return 0

    num_of_bits = range(NUM_OF_BITS)
    real_len = [len(res)] * NUM_OF_BITS
    tows = map(lambda bit: sum(map(lambda x: is_two(x[bit]), res)), num_of_bits)
    total = map(lambda bit: sum(map(lambda x: int(x[bit]), res)), num_of_bits)
    new_len = map(add, tows, real_len)

    probability = list(map(truediv, total, new_len))

    print(total)
    print(probability)

    def I_E(f):
        if f == 0 or f == 1:
            return 0
        return - (f * math.log2(f) + (1 - f) * math.log2(1 - f))

    gains = list(map(I_E, probability))

    print(gains.index(max(gains)))


best_gain({1:'0', 63:'1'})