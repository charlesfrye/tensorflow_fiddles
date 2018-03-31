import tensorflow as tf
import numpy as np

def get_batch(data, batch_size):
    """draw, without replacement, a batch of size batch_size from data
    """

    if hasattr(data, "next_batch"):
        batch_inputs, batch_labels = data.next_batch(batch_size)
    else:
        num_elements = data["labels"].shape[0]
        indices = np.random.choice(num_elements, size=batch_size, replace=False)
        batch_inputs = data["images"][indices,:]
        batch_labels = data["labels"][indices]

    return batch_inputs, batch_labels

def gradient_descent(net, dataset, num_steps, batch_size):
    graph, graph_dict, hyperparameters = net.graph, net.graph_dictionary, net.hyperparameters
    
    initial_parameters = get_random_parameters(hyperparameters)

    input = graph_dict["input"]
    labels = graph_dict["labels"]
    cost_op = graph_dict["cost"]

    gradient_descent_step = graph_dict["step_gradient_descent"]

    gradient_norm_op = graph_dict["gradient_norm"]

    parameters_placeholder = graph_dict["parameters_placeholder"]
    parameters_var = graph_dict["parameters"]

    batch_size=1000

    with tf.Session(graph=graph) as sess:    

        initializer_feed_dict = {parameters_placeholder: initial_parameters}
        tf.global_variables_initializer().run(initializer_feed_dict)
        print("gradient_norm"+"\t"+"cost")
        for step_idx in range(num_steps):
            batch_inputs, batch_labels = get_batch(dataset["train"], batch_size)
            train_feed_dict = {input: batch_inputs,
                               labels: batch_labels}
            _, current_gradient_norm, cost = sess.run([gradient_descent_step, gradient_norm_op, cost_op],
                                                      feed_dict=train_feed_dict)

            if (step_idx % 100) == 0:
                print(current_gradient_norm, cost)
    
        final_gd_parameters = sess.run(parameters_var)
    
    return final_gd_parameters

def optimally_fudged_newton_method(net, dataset, num_steps=10, batch_size=1000,
                                   initial_parameters = None,
                                   fudge_factors = np.logspace(6,-6, num=26, dtype=np.float32)):
    graph, graph_dict, hyperparameters = net.graph, net.graph_dictionary, net.hyperparameters
    
    if initial_parameters is None:
        initial_parameters = get_random_parameters(hyperparameters)
    
    input = graph_dict["input"]
    gradients_op = graph_dict["gradients"]
    labels = graph_dict["labels"]

    gradient_norm_op = graph_dict["gradient_norm"]
    fudge_factor_placeholder = graph_dict["inverse_parameter"]
    inverse_hessian_op = graph_dict["inverse_hessian"]

    parameters_placeholder = graph_dict["parameters_placeholder"]
    parameters_var = graph_dict["parameters"]

    with tf.Session(graph=graph) as sess:    

        initializer_feed_dict = {parameters_placeholder: initial_parameters}
        tf.global_variables_initializer().run(initializer_feed_dict)

        for step_idx in range(num_steps):
            print(step_idx)
            current_parameter_values = sess.run(parameters_var)
            batch_inputs, batch_labels = get_batch(dataset["train"], batch_size)
            train_feed_dict = {input: batch_inputs,
                               labels: batch_labels}
            gradients, current_gradient_norm = sess.run([gradients_op, gradient_norm_op], feed_dict=train_feed_dict)
            print(current_gradient_norm)
            gradient = gradients[0]

            lowest_gradient_norm = current_gradient_norm
            best_parameters = current_parameter_values

            for fudge_factor in fudge_factors:

                reset_feed_dict = {parameters_placeholder: current_parameter_values}
                tf.global_variables_initializer().run(reset_feed_dict)

                train_feed_dict[fudge_factor_placeholder] = fudge_factor
                inverse_hessian = sess.run(inverse_hessian_op, feed_dict=train_feed_dict)
                updated_parameters = current_parameter_values-inverse_hessian.dot(gradient) #do this in TF?

                temp_update_feed_dict =  {parameters_placeholder: updated_parameters}
                tf.global_variables_initializer().run(temp_update_feed_dict)
                new_gradient_norm = sess.run(gradient_norm_op, feed_dict=train_feed_dict)

                if new_gradient_norm < lowest_gradient_norm:
                    print("\t new norm is {0}".format(new_gradient_norm))
                    best_parameters = updated_parameters
                    lowest_gradient_norm = new_gradient_norm

            final_update_feed_dict = {parameters_placeholder: best_parameters}
            tf.global_variables_initializer().run(final_update_feed_dict)
            
        return best_parameters
    
        
def get_random_parameters(hyperparameters):
    num_parameters = hyperparameters["num_parameters"]
    total_weights = hyperparameters["total_weights"]
    total_biases = hyperparameters["total_biases"]
    random_parameters = np.hstack([0.1*np.random.standard_normal(size=total_weights),
                                      [0.1]*total_biases]).astype(np.float32)
    
    return random_parameters