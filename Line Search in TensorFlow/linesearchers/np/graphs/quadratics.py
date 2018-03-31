"""
"""
from collections import namedtuple
import tensorflow as tf
import numpy as np

from . import generic

QuadraticForm = namedtuple("QuadraticForm", ["graph", "graph_dictionary"])

def make(matrix, initial_values,):
    """make a tensorflow graph to evaluate
     0.5*x.transpose().dot(matrix).dot(x)
     the quadratic form on x defined by matrix.

    the input to the matrix is initialized to initial_values.
    """

    assert matrix.shape[0] == matrix.shape[1], "only square matrices can be quadratic forms"
    assert matrix.shape[0] == len(initial_values), "initial_values and matrix must match shape"

    dimension = matrix.shape[0]

    graph = tf.Graph()

    with graph.as_default():
        quadratic_form = tf.constant(matrix, name='quadratic_form')

        parameters_placeholder = tf.placeholder(tf.float32, shape=[dimension],
                                                name="initial_parameters")
        parameters_var = tf.Variable(initial_value=parameters_placeholder,
                                         name="parameters_variable",
                                    expected_shape=[dimension])

        parameters_vector = tf.reshape(parameters_var, (dimension, 1))

        output = 0.5*tf.squeeze(tf.matmul(parameters_vector,
                                  tf.matmul(quadratic_form, parameters_vector),
                                      transpose_a=True,
                                  name='output'),
                        name='squeezed_output')

        graph_dictionary = {"parameters_placeholder": parameters_placeholder,
                            "parameters": parameters_var,
                           "output": output,}

        generic.add_gradient_ops(output, parameters_var, graph_dictionary)
        generic.add_hess_ops(output, parameters_var, graph_dictionary)

    return QuadraticForm(graph, graph_dictionary)

## convenience functions for interacting with quadratic form graphs

def generate_initial_values(N, scaling_factor=None):
    """generate a vector of scaled random normal values.
    scaling_factor defaults to 1/sqrt(N).
    """
    if scaling_factor is None:
        scaling_factor = 1/np.sqrt(N)

    return scaling_factor*np.random.standard_normal(size=N).astype(np.float32)

def run_algorithm(quadratic_form, algorithm, num_steps):
    """run algorithm on quadratic form and return the resulting outputs and values
    """
    graph, graph_dictionary = quadratic_form

    with graph.as_default():
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            graph_dictionary["parameters"].initializer.run()
            for _ in range(num_steps):
                sess.run(graph_dictionary["step_"+algorithm])
            output = sess.run(graph_dictionary["output"])
            values = graph_dictionary["parameters"].eval()

    return output, values

def get_result(result_key, parameters_vector, quadratic_form):
    """return the result of applying quadratic_form.graph_dictionary[result_key] on parameters_vector
    """

    graph, graph_dictionary = quadratic_form
    result_op = graph_dictionary[result_key]
    parameters_placeholder = graph_dictionary["parameters_placeholder"]

    parameters_feed_dict = make_feed_dict(parameters_placeholder, parameters_vector)

    result = run_op(graph, result_op, parameters_feed_dict)

    return result

def run_op(graph, op, parameters_feed_dict):
    """run a given op on a given graph with feed_dict input_feed_dict
    """

    with graph.as_default():
        with tf.Session() as sess:
            tf.global_variables_initializer().run(parameters_feed_dict)
            result = sess.run(op)

    return result

def make_feed_dict(parameters_placeholder, parameters_vector):
    """make a feed_dict for a quadratic_form_graph
    """

    feed_dict = {parameters_placeholder:parameters_vector}

    return feed_dict

## functions for generating various random matrices

def generate_gaussian(N):
    """generate an N by N gaussian random matrix with variance N
    """
    return 1/np.sqrt(N)*np.random.standard_normal(size=(N,N)).astype(np.float32)

def generate_symmetric(N):
    """generate an N by N symmetric gaussian random matrix with variance N
    """
    base_matrix = generate_gaussian(N)
    return (1/np.sqrt(2))*(base_matrix+base_matrix.T)

def generate_wishart(N, k=1):
    """generate an N by N wishart random matrix with rank min(N,k)
    """
    self_outer_product = lambda x: x.dot(x.T)
    wishart_random_matrix = 1/k*self_outer_product(np.random.standard_normal(size=(N,k))).astype(np.float32)

    return wishart_random_matrix

def generate_negative_wishart(N, k=1):
    """generate an N by N negative wishart random matric with rank min(N,k)
    """
    wishart_random_matrix = generate_wishart(N, k)
    negative_wishart_random_matrix = -1*wishart_random_matrix

    return negative_wishart_random_matrix