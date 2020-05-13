import logging
import networkx as nx
import matplotlib.pyplot as plt
import HW1.rulesDecisionTree as rdt

from xgboost import XGBRegressor
from HW1 import utils, packetClassifier
from sklearn.ensemble import RandomForestRegressor


def exercise1_3_1(groups):
    # logging.basicConfig(filename="exercise1_3_1_{groups}.log".format(groups=groups), level=logging.INFO)
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        logging.info("Choose bit by IG , minimum subgroups size " + str(groups))
        decision_tree = rdt.create_decision_tree(rules_df, min_group_count, utils.best_bit_by_IG)
        plot(decision_tree, "{folder}DT_1_3_1_{mgc}".format(folder="HW1/output/", mgc=min_group_count))


def exercise1_3_2(groups):
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        decision_tree = rdt.create_decision_tree(rules_df, min_group_count, None)
        plot(decision_tree, "{folder}DT_1_3_2_{mgc}".format(folder="HW1/output/", mgc=min_group_count))


def exercise2_3_1(groups):
    rules_df = utils.create_rule_table()
    for min_group_count in groups:
        decision_tree = rdt.create_decision_tree(rules_df, min_group_count, utils.best_bit_by_entropy)
        plot(decision_tree, "{folder}DT_2_{mgc}".format(folder="HW1/output/", mgc=min_group_count))


def exercise3():
    packets_df = utils.create_packet_table(400000)
    packets_df, NAlist = utils.reduce_mem_usage(packets_df)
    packetClassifier.classify_packets(packets_df, RandomForestRegressor(random_state=1))


def exercise4():
    packets_df = utils.create_packet_table(400000)
    packets_df, NAlist = utils.reduce_mem_usage(packets_df)
    packetClassifier.classify_packets(packets_df, XGBRegressor())


def plot(decision_tree, save_file_path):
    nx.write_adjlist(decision_tree, save_file_path)
    logging.info("Number of nodes: {}".format(decision_tree.number_of_nodes()) + "\n")
    logging.info("Max depth: {}".format(nx.algorithms.dag.dag_longest_path_length(decision_tree)) + "\n")
    logging.info("longest path: {}".format(nx.algorithms.dag.dag_longest_path(decision_tree)) + "\n")
    # nx.draw(decision_tree, with_labels=True)
    plt.show()


def main():
    groups = [128, 96, 64, 32, 16]
    # groups = [300]
    # logging.info('minimum groups size: {}'.format(groups))
    # exercise1_3_1(groups)
    # exercise1_3_2(groups)
    exercise2_3_1(groups)
    # exercise3()
    # exercise4()

