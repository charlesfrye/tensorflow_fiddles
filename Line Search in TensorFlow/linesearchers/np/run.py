import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from . import graphs

def btls(matrix, initial_values, beta, alpha, method=None, num_steps=1, max_backtrack=10, minimum_eigenvalue=None):
    qf = graphs.quadratics.make(matrix, initial_values)
    parameters_placeholder = qf.graph_dictionary["parameters_placeholder"]
    output = qf.graph_dictionary["output"]
    gradients = qf.graph_dictionary["gradients"]
    hessian_matrix = qf.graph_dictionary["hessian_matrix"]

    with tf.Session(graph=qf.graph) as sess:

        parameter_values = np.copy(initial_values)

        parameters_feed_dict =  graphs.quadratics.make_feed_dict(parameters_placeholder, parameter_values)
        tf.global_variables_initializer().run(parameters_feed_dict)

        initial_output = sess.run(output)

        for step_idx in range(num_steps):
            parameters_feed_dict =  graphs.quadratics.make_feed_dict(parameters_placeholder, parameter_values)
            tf.global_variables_initializer().run(parameters_feed_dict)

            gradient = sess.run(gradients)[0]
            if method in ["newton", "nc_newton"]:
                hessian = sess.run(hessian_matrix)
            else:
                hessian = None
                
            search_direction = get_search_direction(method, gradient, hessian, minimum_eigenvalue)
            
            parameter_values, sufficient_decrease = do_backtrack(sess, output, gradients, parameters_placeholder,
                                                                  parameter_values, search_direction,
                                                                  alpha, beta, max_backtrack)
            if not sufficient_decrease:
                break

        parameters_feed_dict =  graphs.quadratics.make_feed_dict(parameters_placeholder, parameter_values)
        tf.global_variables_initializer().run(parameters_feed_dict)
        final_output = sess.run(output)
        
    results = {"steps_run":step_idx+1,
               "sufficient_decrease":sufficient_decrease,
               "initial_values": initial_values,
               "solution": parameter_values,
               "initial_f": initial_output,
               "final_f": final_output}
    
    return results

def decrease_is_sufficient(f_init, f_val, sufficient_decrease_criterion_slope, eta, update):
    """armijo rule: has function gone down by at least some fraction of the amount predicted
    by the local linear approximation?
    """
    return f_val <= f_init + eta*sufficient_decrease_criterion_slope.dot(update)

def do_backtrack(sess, op, gradients_op, parameters_placeholder,
                  parameter_values, search_direction,
                  alpha, beta, max_backtrack=10):
    gradients, initial_op_value = sess.run([gradients_op, op])
    
    sufficient_decrease_criterion_slope = alpha*gradients[0]
    etas = [beta**k for k in range(max_backtrack+1)]
    sufficient_decrease = False
    for eta in etas:
        update = eta*search_direction
        updated_parameters = parameter_values+update

        parameters_feed_dict =  graphs.quadratics.make_feed_dict(parameters_placeholder, updated_parameters)
        tf.global_variables_initializer().run(parameters_feed_dict)

        op_value_at_updated_parameters = sess.run(op)

        if decrease_is_sufficient(initial_op_value, op_value_at_updated_parameters,
                                      sufficient_decrease_criterion_slope, eta, update):
            sufficient_decrease = True
            break
                        
    if not sufficient_decrease:
        updated_parameters = parameter_values
    
    return updated_parameters, sufficient_decrease

def get_search_direction(method, gradient, hessian=None, minimum_eigenvalue=None):
    if method is "gradient_descent":
        return -gradient
    
    elif method is "newton":
        assert hessian is not None, "must provide hessian to use {0}".format(method)
        inverse_hessian = np.linalg.inv(hessian)
        newton_direction = -inverse_hessian.dot(gradient)
        return newton_direction
    
    elif method is "nc_newton":
        assert hessian is not None, "must provide hessian to use {0}".format(method)
        assert minimum_eigenvalue is not None, "must provide minimum_eigenvalue to use {0}".format(method)
        PT_inverse_hessian = compute_PT_inverse(hessian, minimum_eigenvalue)
        nc_newton_direction = -PT_inverse_hessian.dot(gradient)
        return nc_newton_direction
    
    else:
        raise NotImplementedError("no method for {0}".format(method))
        
def compute_PT_inverse(matrix, minimum_eigenvalue):
    """diagonalize, set eigenvalues below minimum_eigenvalue to minimum_eigenvalue,
    then return to original basis
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    truncated_eigenvalues = np.where(np.abs(eigenvalues)>=minimum_eigenvalue, np.abs(eigenvalues), minimum_eigenvalue)

    inverted_eigenvalues = np.divide(1.0, truncated_eigenvalues)

    rescaled_eigenvectors = np.multiply(inverted_eigenvalues, eigenvectors)

    PT_inverse = eigenvectors.dot(rescaled_eigenvectors.T)
    
    return PT_inverse

def plot_results(results):
    coords = np.asarray([result["initial_values"] for result in results])
    satisfied_condition = [result["sufficient_decrease"] for result in results]
    
    plt.scatter(coords[:,0], coords[:,1], c=satisfied_condition, cmap="RdBu", vmin=0, vmax=1)
    plt.colorbar(ticks=[0,1])
    plt.hlines(0, -5,5, linewidth=2)
    plt.vlines(0, -5,5, linewidth=2)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.axis("off")