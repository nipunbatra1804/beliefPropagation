# Matric Number: A0098162Y
# Please add comments to explain your logic, wherever necessary.

import numpy as np
from functools import reduce
from tree import Tree, TreeNode


def sum_product(tree: Tree, observations: dict) -> dict:
    """Implementation of the Sum-Product algorithm for directed trees.

    Arguments:
        tree {Tree} -- A Tree object.
        observations {dict} -- A dict of observations where keys are node ids
                               and values are the observed values of the nodes.
                               For example, {"0": 1, "2": 0} means nodes 0 and 2
                               have the values 1 and 0 respectively.
    Returns:
        dict -- A dict of marginals for all the nodes. The keys of this dict are
                node ids and the values are numpy arrays of shape (1, dim) where
                dim is the dimension of the node.
    """

    setup_self_factors(tree, observations)
    root = tree.root
    for c in root.children:
        collect(tree, root, c)
    for c in root.children:
        distribute(tree, root, c)
    return compute_marginals(tree)


def setup_self_factors(tree: Tree, observations: dict) -> None:
    """This function sets up the self-factors for each node and assigns
       these values to tree.self_factors for each node.

    Arguments:
        tree {Tree} -- A Tree object.
        observations {dict} -- A dict of observations where keys are node ids
                               and values are the observed values of the nodes.
                               For example, {"0": 1, "2": 0} means nodes 0 and 2
                               have the values 1 and 0 respectively.
    """
    for key, value in tree.nodes.items():
        tree.self_factors[value.id] = np.ones(value.dim)
        if(str(value.id) in observations.keys()):
            deltaArray = np.zeros(value.dim)
            observationId = str(value.id)
            deltaArray[observations[observationId]] = 1
            tree.self_factors[value.id] = deltaArray
    return None


def collect(tree: Tree, to_node: TreeNode, from_node: TreeNode) -> None:
    """This function collects messages from child node to parent node.

    Arguments:
        tree {Tree} -- A Tree object.
        to_node {TreeNode} -- The parent node to which the message is sent.
        from_node {TreeNode} -- The child node from which the message is sent.
    """
    for childNode in from_node.children:
        collect(tree, from_node, childNode)
    send_message(tree, from_node, to_node)


def distribute(tree: Tree, from_node: TreeNode, to_node: TreeNode) -> None:
    """This function distributes messages from parent node to child node.

    Arguments:
        tree {Tree} -- A Tree object.
        from_node {TreeNode} -- The parent node from which the message is sent.
        to_node {TreeNode} -- The child node to which the message is sent.
    """
    send_message(tree, from_node, to_node)
    for childNode in to_node.children:
        distribute(tree, to_node, childNode)


def get_evidence_potential(factor_table, node_self_factor):
    # return phi(xi,xj) or delta(xj)phi(xi,xj)

    if(np.all(node_self_factor == 1)):
        return factor_table
    evidence_potential = np.ones(factor_table.shape)
    for x in range(0, node_self_factor.shape[0]):
        evidence_potential[:, x] = factor_table[:, x]*node_self_factor[x]

    return evidence_potential


def send_message(tree: Tree, from_node: TreeNode, to_node: TreeNode) -> None:
    """This function sends a message from from_node to to_node. This function
       assumes that all the messages required to send a message from from_node
       to to_node have already been cached in tree.messages.

       Upon completion, this function doesn't return anything but caches the
       message from from_node to to_node in tree.messages.

    Arguments:
        tree {Tree} -- A Tree object.
        from_node {TreeNode} -- A TreeNode object from which the message is
                                is being sent.
        to_node {TreeNode} -- A TreeNode object to which the message is
                                is being sent.
    """

    # construct message id to add to dictinary
    messageId = '{fr}-{to}'.format(to=to_node.id, fr=from_node.id)
    print("computing message for ", messageId)

    # construct factor id to retrieve factor table from Tree object
    factorId = '{to}-{fr}'.format(to=to_node.id, fr=from_node.id)
    factor_table = None

    # get transposed factor matrix if corresponding factor is not found in tree
    if(factorId not in tree.factors):
        factorId = '{fr}-{to}'.format(to=to_node.id, fr=from_node.id)
        factor_table = np.transpose(tree.factors[factorId])
    else:
        factor_table = tree.factors[factorId]

    node_self_factor = tree.self_factors[from_node.id]

    # calculate phi_e(xj) * phi(xi,xj)
    evidence_potential = get_evidence_potential(
        factor_table, node_self_factor)

    prod_of_messages = 1

    # if the root node is present use it as self factor
    if(from_node.id == 0):
        prod_of_messages = tree.p_0

    # get neighbour nodes for the from_node
    neighborNodes = from_node.children + [from_node.parent]
    for node in neighborNodes:
        if(node is None):
            continue
        # make sure to exclude the to_node
        if(node.id == to_node.id):
            continue
        message_key = '{fr}-{to}'.format(to=from_node.id, fr=node.id)
        if(message_key in tree.messages.keys()):
            prod_of_messages = np.multiply(
                prod_of_messages, tree.messages[message_key])

    # multiply potentials of i,j with incoming messages
    product_of_potentials = np.dot(
        evidence_potential, np.transpose(prod_of_messages))

    # sum up over xj
    tree.messages[messageId] = np.transpose(
        np.sum(product_of_potentials, axis=1, keepdims=True))


def compute_marginals(tree: Tree) -> dict:
    """This function computes the marginals of all nodes in the tree
       once all the messages have been cached in tree.messages.

       For example, for the following tree with all nodes representing
       binary random variables, this function will return:
       {0: p(x0),
        1: p(x1),
        2: p(x2)}

                0 ----> 1
                |
                ------> 2

       For the following tree with all nodes representing
       a binary random variables and the observation {"3": 1}, this function
       will return:
       {0: p(x0 | x3 = 1),
        1: p(x1 | x3 = 1),
        2: p(x2 | x3 = 1),
        3: p(x3 | x3 = 1)}

                0 ----> 1 ----> 3
                |
                ------> 2

        Since x3 is observed in this case, p(x3 | x3 = 1) is equal to
        np.array([[0., 1.]]).

    Arguments:
        tree {Tree} -- A Tree object.

    Returns:
        dict -- A dict of marginals for all the nodes. The keys of this dict are
                node ids and the values are numpy arrays of shape (1, dim) where
                dim is the dimension of the node.
    """

    marginals = {}
    for key, node in tree.nodes.items():
        neighborNodes = node.children + [node.parent]
        prod_of_messages = 1
        for neighbor in neighborNodes:
            if(neighbor is None):
                continue
            inwardMessageKey = '{fr}-{to}'.format(
                to=node.id, fr=neighbor.id)
            if(inwardMessageKey in tree.messages.keys()):
                prod_of_messages = np.multiply(
                    prod_of_messages, tree.messages[inwardMessageKey])

        # include prior for the root node and multiply self_factor with the node.
        # for sake of convenience, we just multiply the prior along with the self factors if the nod eis a root node.
        if (node.id == 0):
            prod_of_messages = np.multiply(prod_of_messages, tree.p_0)
        prod_of_messages = np.multiply(
            prod_of_messages, tree.self_factors[node.id])

        # normalize marginal over sum
        marginal = prod_of_messages / np.sum(prod_of_messages)

        marginals[node.id] = marginal

    return marginals
