import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import HW1.rulesDecisionTree as rdt

from itertools import tee
from xgboost import XGBRegressor
from HW1 import utils, packetClassifier
from sklearn.ensemble import RandomForestRegressor


def exercise1_3_1(groups):
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        logging.info("exercise1_3_1 : min sub group %s" % min_group_count)
        decision_tree = rdt.create_decision_tree(rules_df, min_group_count, utils.best_bit_by_IG)
        plot(decision_tree, "{folder}DT_1_3_1_{mgc}".format(folder="HW1/output/", mgc=min_group_count))


def exercise1_3_2(groups):
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        decision_tree = rdt.create_decision_tree(rules_df, min_group_count, None)
        plot(decision_tree, "{folder}DT_1_3_2_{mgc}".format(folder="HW1/output/", mgc=min_group_count))


def exercise2(groups):
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        decision_tree = rdt.create_decision_tree(rules_df, min_group_count, utils.best_bit_by_entropy)
        plot(decision_tree, "{folder}DT_2_{mgc}".format(folder="output/", mgc=min_group_count))


def exercise2_3_2(groups):
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        decision_tree, _ = rdt.create_tree(rules_df, min_group_count, utils.best_bit_by_entropy, True)
        plot(decision_tree, "{folder}DT_2_3_2_{mgc}".format(folder="output/", mgc=min_group_count))


def exercise3():
    packets_df = utils.create_packet_table(400000)
    packets_df, NAlist = utils.reduce_mem_usage(packets_df)
    packetClassifier.classify_packets(packets_df, RandomForestRegressor(random_state=1))


def exercise4():
    packets_df = utils.create_packet_table(400000)
    packets_df, NAlist = utils.reduce_mem_usage(packets_df)
    packetClassifier.classify_packets(packets_df, XGBRegressor())


def exercise2_3_2(groups):
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        decision_tree, _ = rdt.create_tree(rules_df, min_group_count, utils.best_bit_by_entropy, True)
        plot(decision_tree, "{folder}DT_2_3_2_{mgc}".format(folder="output/", mgc=min_group_count))


def plot(decision_tree, save_file_path):
    nx.write_adjlist(decision_tree, save_file_path)
    dists1, dists2, dists3 = tee(utils.dist_to_leaves(decision_tree), 3)
    logging.info("Number of nodes: {}".format(decision_tree.number_of_nodes()))
    logging.info("Max depth: {}".format(max(dists1)))
    logging.info("Min depth: {}".format(min(dists2)))
    logging.info("Mean depth: {}".format(np.fromiter(dists3, int).mean()))
    logging.info("longest path: {}".format(nx.algorithms.dag.dag_longest_path(decision_tree)))
    # nx.draw(decision_tree, with_labels=True)
    plt.show()


def main():
    groups = [128, 96, 64, 32, 16]
    # groups = [16]
    # logging.info('minimum groups size: {}'.format(groups))
    # exercise1_3_1(groups)
    # exercise1_3_2(groups)
    # exercise2_3_1(groups)
    # exercise2_3_2(groups)
    exercise3(200000)
    # exercise4(400000)

