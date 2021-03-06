# Important: must run process_data.py first!
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

num_iter = 200
learn_rate = 0.01

# Train/fit
training_data = pd.read_csv("datasets/project_training.csv")
inputs = training_data.loc[:, "4/1/20":"4/7/20"].values.tolist()
true_vals = training_data["w2inc"].tolist()
weights = np.random.uniform(-0.05, 0.05, len(inputs[0]))
bias = random.uniform(-0.05, 0.05)

for _ in range(num_iter):
    # Calculate output and update weights for each input
    for inpt, true_val in zip(inputs, true_vals):
        output = bias
        # Calculate output based on weights
        for i in range(len(weights)):
            output += weights[i] * inpt[i]
        if output >= 0:
            output = 1
        else:
            output = 0
        # Update weights based on output
        for i in range(len(weights)):
            weights[i] += learn_rate * (true_val - output) * inpt[i]
        # Update bias weight
        bias += learn_rate * (true_val - output)
# Predict
testing_data = pd.read_csv("datasets/project_test.csv")
inputs = testing_data.loc[:, "4/8/20":"4/14/20"].values.tolist()
predictions = []
# Calculate output for each input
for inpt in inputs:
    output = bias
    for i in range(len(weights)):
        output += weights[i] * inpt[i]
    if output >= 0:
        output = 1
    else:
        output = 0
    predictions.append(output)

testing_data["pred"] = predictions
testing_data.to_csv("predictions_perceptron.csv", index=False)

# Calculate metrics
tp = 0
fp = 0
fn = 0
tn = 0
for pred, actual in zip(predictions, testing_data["w3inc"]):
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
print("Youden index: {:.4f}".format(tp / (fn + tp) - fp / (tn + fp)))
print("Confusion matrix:\n{} | {}\n{} | {}".format(tn, fp, fn, tp))
print("TPR: {:.4f}".format(tp / (fn + tp)))
print("FPR: {:.4f}".format(fp / (tn + fp)))
print("TNR: {:.4f}".format(tn / (tn + fp)))
print("FNR: {:.4f}".format(fn / (fn + tp)))
print("PPV: {:.4f}".format(tp / (tp + fp)))
print("NPV: {:.4f}".format(tn / (tn + fn)))