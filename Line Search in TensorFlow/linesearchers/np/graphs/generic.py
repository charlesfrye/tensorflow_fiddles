"""functions for adding optimizers, crit_finders, and other ops to generic graphs
"""
import tensorflow as tf

def add_gradient_ops(function, inputs, graph_dictionary):
    """adds ops to calculate gradients and scaled squared gradient norm to graph and graph_dictionary
    to graph, calculates squared gradient norm, and adds these operations
    to the graph_dictionary
    """
    with tf.variable_scope("gradients"):

        gradients = tf.gradients(function, inputs, name="gradients")
        gradient_norm = tf.norm(gradients, name="gradient_norm")
        scaled_squared_gradient_norm = 0.5*tf.square(gradient_norm, name="scaled_squared_gradient_norm")

    graph_dictionary.update({
                           "gradients": gradients,
                           "gradient_norm": gradient_norm,
                           "scaled_squared_gradient_norm": scaled_squared_gradient_norm
                           })

def add_hess_ops(function, inputs, graph_dictionary):
    """adds ops to calculate and diagonalize the hessian to the graph and graph_dictionary
    """
    with tf.variable_scope("hessian"):
        hessian_matrix = tf.hessians(function, inputs, name="hessians_output")[0]
        eigenvalues, eigenvectors = tf.self_adjoint_eig(hessian_matrix)

    graph_dictionary.update({
                           "hessian_matrix": hessian_matrix,
                           "eigenvalues": eigenvalues,
                           "eigenvectors": eigenvectors
                           })

def add_optimizer(function, inputs, hyperparameters, graph_dictionary):
    """adds ops to optimize function according to hyperparameters to the graph and graph_dictionary
    """
    with tf.variable_scope("optimizer"):
        optimizer_step_ct = tf.Variable(0, trainable=False)
        optimizer_rate = hyperparameters["learning_rate"]

        if "optimizer_decay_rate" in hyperparameters.keys():
            assert "optimizer_decay_every" in hyperparameters.keys(), "missing decay_steps for gradient_descent"
            optimizer_rate =  tf.train.exponential_decay(optimizer_rate,
                                                        optimizer_step_ct,
                                                        decay_steps=hyperparameters["gradient_descent_decay_every"],
                                                        decay_rate=hyperparameters["gradient_descent_decay_rate"])

        if "momentum_rate" in hyperparameters.keys():
            optimizer = tf.train.MomentumOptimizer(optimizer_rate, hyperparameters["momentum_rate"])
            step_optimizer = optimizer.minimize(function)
            graph_dictionary["step_momentum"] = step_optimizer
        else:
            optimizer = tf.train.GradientDescentOptimizer(optimizer_rate)
            step_optimizer = optimizer.minimize(function)
            graph_dictionary["step_gradient_descent"] = step_optimizer