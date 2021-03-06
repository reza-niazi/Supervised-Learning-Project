# Important: must run process_data.py first!
import pandas as pd
import numpy as np
from math import exp
import random
import matplotlib.pyplot as plt

learn_rate = 0.01
num_iter = 200
rand_seed = 1234

# Create neural network with 1 hidden layer
def create_network(num_in, num_hidden, num_out):
    net = {}
    hidden_neurons = []
    for _ in range(num_hidden):
        neuron = {}
        neuron["weights"] = np.random.uniform(-0.05, 0.05, num_in)
        neuron["bias"] = random.uniform(-0.05, 0.05)
        hidden_neurons.append(neuron)
    net["hidden"] = hidden_neurons
    output_neurons = []
    for _ in range(num_out):
        neuron = {}
        neuron["weights"] = np.random.uniform(-0.05, 0.05, num_hidden)
        neuron["bias"] = random.uniform(-0.05, 0.05)
        output_neurons.append(neuron)
    net["output"] = output_neurons
    return net

# Computes the output of a neuron using the given inputs
def compute_output(neuron, inputs):
    output = neuron["bias"]
    for i in range(len(inputs)):
        output += neuron["weights"][i] * inputs[i]
    return output

# Sigmoid activation function
def activation(value):
    # Prevents error from overflow when value is large and negative
    try:
        return 1 / (1 + exp(-value))
    except OverflowError:
        return 0

# Calculates the output for each neuron in the network
def forward_propagate(net, inputs):
    outputs_hidden = []
    for neuron in net["hidden"]:
        outputs_hidden.append(activation(compute_output(neuron, inputs)))
    outputs = []
    for neuron in net["output"]:
        outputs.append(activation(compute_output(neuron, outputs_hidden)))
    return outputs_hidden, outputs

# Calculates error for a neuron in the output layer
def error_output(output, true_val):
    return output * (1 - output) * (true_val - output)

# Calculates error for a neuron in the hidden layer
def error_hidden(net, index, hidden_output, output_errors):
    total = 0
    for i, error in enumerate(output_errors):
        total += net["output"][i]["weights"][index] * error
    return hidden_output * (1 - hidden_output) * total

# Backpropagates errors to the hidden layer
def back_propagate(net, outputs, outputs_hidden, true_vals):
    output_errors = []
    for i in range(len(outputs)):
        output_errors.append(error_output(outputs[i], true_vals[i]))

    hidden_errors = []
    for i in range(len(outputs_hidden)):
        hidden_errors.append(error_hidden(net, i, outputs_hidden[i], output_errors))

    return hidden_errors, output_errors

# Updates weights for each neuron based on errors
def update_weights(net, hidden_errors, output_errors, inputs):
    for i in range(len(net["hidden"])):
        neuron = net["hidden"][i]
        error = hidden_errors[i]
        for j in range(len(neuron["weights"])):
            neuron["weights"][j] += learn_rate * error * inputs[j]
        neuron["bias"] += learn_rate * error

    for i in range(len(net["output"])):
        neuron = net["output"][i]
        error = output_errors[i]
        for j in range(len(neuron["weights"])):
            neuron["weights"][j] += learn_rate * error * outputs_hidden[j]
        neuron["bias"] += learn_rate * error

# hidden_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hidden_nums = [8]
np.random.seed(rand_seed)
random.seed(rand_seed)
f = open("results.csv", "a")
f.write("Hidden layer size,TN,FP,FN,TP,TPR,FPR,TNR,FNR,PPV,NPV\n")
for num_hidden in hidden_nums:
    # Create neural network
    net = create_network(7, num_hidden, 1)

    # Train/fit
    training_data = pd.read_csv("datasets/project_training.csv")
    inputs = training_data.loc[:, "4/1/20":"4/7/20"].values.tolist()
    # inputs = training_data.loc[:, "0":"6"].values.tolist()
    true_vals = [[value] for value in training_data["w2inc"]]
    # true_vals = [[value] for value in training_data["inc"]]
    iters = 0
    while(iters < num_iter):
        # Based on backpropagation algorithm from notes on Canvas (page 4 of Neural_Networks.pdf)
        for inpt, true_val in zip(inputs, true_vals):
            # Step 1
            outputs_hidden, outputs = forward_propagate(net, inpt)
            # Step 2 and 3
            hidden_errors, output_errors = back_propagate(net, outputs, outputs_hidden, true_val)
            # Step 4
            update_weights(net, hidden_errors, output_errors, inpt)
        iters += 1
        print("Iteration {}/{} done".format(iters, num_iter))
        
    # Predict
    testing_data = pd.read_csv("datasets/project_validation.csv")
    inputs = testing_data.loc[:, "4/8/20":"4/14/20"].values.tolist()
    # inputs = testing_data.loc[:, "0":"6"].values.tolist()
    predictions = []
    # Calculate output for each input
    for inpt in inputs:
        outputs_hidden, outputs = forward_propagate(net, inpt)
        predictions.append(outputs[0])

    testing_data["pred"] = predictions
    testing_data.to_csv("predictions_nn.csv", index=False)

    # Calculate metrics and plot ROC curve

    print("Calculating metrics for neural network with 1 hidden layer of {} neurons".format(num_hidden))
    # Count number of positive and negative occurences
    num_p = 0
    num_n = 0
    for value in testing_data["w3inc"]:
    # for value in testing_data["inc"]:
        if value == 1:
            num_p += 1
        else:
            num_n += 1

    tpr = []
    fpr = []
    youden_index = -1
    # For 1001 different probability thresholds 
    for i in range(1001):
        tp = 0
        fp = 0 
        binary_pred = [0 if predictions[j] < (i / 1000) else 1 for j in range(len(predictions))]
        for pred, actual in zip(binary_pred, testing_data["w3inc"]):
        # for pred, actual in zip(binary_pred, testing_data["inc"]):
            if actual == 1:
                if pred == 1:
                    tp += 1
            else:
                if pred == 1:
                    fp +=1
        # Calculate TPR and FPR for each threshold
        tpr.append(tp / num_p)
        fpr.append(fp / num_n)
        if (tp / num_p - fp / num_n) > youden_index:
            youden_index = tp / num_p - fp / num_n
            best_i = i
    print("Highest Youden index: {:.4f} at threshold of {}".format(youden_index, best_i / 1000))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    binary_pred = [0 if predictions[j] < (best_i / 1000) else 1 for j in range(len(predictions))]
    for pred, actual in zip(binary_pred, testing_data["w3inc"]):
    # for pred, actual in zip(binary_pred, testing_data["inc"]):
        if actual == 1:
            if pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred == 1:
                fp +=1
            else:
                tn += 1
    print("Confusion matrix:\n{} | {}\n{} | {}".format(tn, fp, fn, tp))
    print("TPR: {:.4f}".format(tp / (fn + tp)))
    print("FPR: {:.4f}".format(fp / (tn + fp)))
    print("TNR: {:.4f}".format(tn / (tn + fp)))
    print("FNR: {:.4f}".format(fn / (fn + tp)))
    print("PPV: {:.4f}".format(tp / (tp + fp)))
    print("NPV: {:.4f}".format(tn / (tn + fn)))
    f.write("{}, {},{},{},{},{},{},{},{},{},{}\n".format(num_hidden,tn,fp,fn,tp,tp / (fn + tp),fp / (tn + fp),tn / (tn + fp),fn / (fn + tp),tp / (tp + fp),tn / (tn + fn)))

    plt.figure()
    # Dashed grey line for random predictor
    plt.plot([0, 1], [0, 1], color="grey", linestyle="dashed", label="Random classifier")
    # Plot FPR and TPR
    plt.plot(fpr, tpr, marker='o', label="Neural network")
    # plt.title("Neural Network ROC Curve (hidden layer size of {})".format(num_hidden))
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    # plt.legend(loc="lower right")
f.close()
plt.show()