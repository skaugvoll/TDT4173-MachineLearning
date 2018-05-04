""" Convolutional Neural Network.

This script follows TensorFlow layers API, see 'convolutional_network_raw'
example for a raw implementation with variables.

"""
from data_generator import DataGenerator
import numpy as np
import tensorflow as tf


def _init_global_problem_specific_parameters(img_height, img_width):
    global pic_height, pic_width
    pic_height = img_height
    pic_width = img_width


def _init_global_training_parameters(lr, steps, batch):
    global learning_rate, num_steps, batch_size
    learning_rate = lr
    num_steps = steps
    batch_size = batch

def _init_global_network_parameters(num_features, num_labels, dropProb):
    global num_input, num_classes, dropout
    num_input = num_features
    num_classes = num_labels
    dropout = dropProb

def init_global_parameters(img_height, img_width, features):
    if (features == None):
        features = img_height * img_width


    # Problem Specific Parameters
    # CHAR74 has pictures where ALL of them have the same dimensions 20x20pixels
    pic_height = img_height
    pic_width = img_width
    _init_global_problem_specific_parameters(pic_height, pic_width)

    # Training Parameters
    learning_rate = 0.0015 #0.001
    num_steps = 500 #2000
    batch_size = 50
    _init_global_training_parameters(learning_rate, num_steps, batch_size)

    # Network Parameters
    num_input = features  # CHAR74 data input (img shape: 20*20 = 400)
    num_classes = 26  # CHAR74 total classes (a-z chars = 26)
    dropout = 0.5  # Dropout, probability to drop a unit
    _init_global_network_parameters(num_input, num_classes, dropout)


def change_num_input(features):
    global num_input
    num_input = features


def change_parameters(*args):
    parameters = ["pic_height", "pic_width", "num_input", "num_classes"]
    for idx, arg in enumerate(*args):
        print("ARG: ", arg)
        if (arg == "" or arg == None):
            continue

        par = parameters[idx]
        eval(par + " = " + str(arg))





# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # CHAR74 data input is a 1-D vector of 400 features (20*20 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, pic_height, pic_width, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "class_ids": pred_classes[:, tf.newaxis],
            "probabilities": pred_probas,
            "logits": logits_test
        }
        es = tf.estimator.EstimatorSpec(mode, predictions=predictions)
        return es


    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


def init_model(img_height=20, img_width=20, features=None):
    init_global_parameters(img_height, img_width, features)
    model = tf.estimator.Estimator(model_fn)

    return model


def train_conv_net(model, training_cases, training_labels):
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': training_cases}, y=training_labels,
                                                  batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)
    return model

def test_conv_net(model, testing_cases, testing_labels):
    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': testing_cases}, y=testing_labels, batch_size=batch_size,
                                                  shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)

    # get the accuracy from the Estimator
    accuracy = e['accuracy']

    return model, accuracy


def prediction_conv_net(model, testing_case):
    # Define the input function for PREDICTION
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': testing_case}, batch_size=batch_size,
                                                  shuffle=False)
    # Use the Estimator 'prediction' method
    e = model.predict(input_fn) # e is a generator object now. now caluclations are done, but it knows what to do..

    # use the generator to generate the class_ids, probabilities and logits

    # print(list(e))
    # res = next(e)

    # print("Pred-Estimator:\n{}\n".format(res))

    # get the probabilities
    # prob = res['probabilities']
    # classes = res['class_ids']

    return e



def run_char74():
    # Create | import the data
    dg = DataGenerator(dataset="../chars74k-lite", normalized=True, threshold=None)
    dg.shuffle_data()

    # extract the training and testing data from our data-generator
    training_cases, training_labels = dg.get_training_partition(percentage=.8)
    testing_cases, testing_labels = dg.get_testing_partition(percentage=.2)

    # convert training and testing data from float64 to float32
    training_cases = training_cases.astype(np.float32)
    testing_cases = testing_cases.astype(np.float32)

    # init global variables, instead of passing as parameters. - because cleaner and more readability
    init_global_parameters(20, 20, 400)


    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    # Train the model
    print("Now training the model with:\nLR: {}\n#Steps: {}\nBatch Size: {}\n".format(learning_rate, num_steps, batch_size))
    model = train_conv_net(model, training_cases, training_labels)

    # Test the model
    print("Done training, now testing the model")
    _, accuracy = test_conv_net(model, testing_cases, testing_labels)

    # print accuracy of the model
    print("Testing Accuracy: {:.2f}%\n".format(accuracy * 100))


    # Prediction by the model, predict the first test case
    # if more than one case to predict, use estimator to get results for each
    print("now prediction using the model")

    print("blaa", testing_cases[:1].shape, testing_labels[:1])
    prob = prediction_conv_net(model, testing_cases[:1])
    print("Predictions probabilities:\n{}\n".format(prob))
    # print("Predictions probabilities:\n{}\n".format(classes))


if __name__ == "__main__":
    run_char74()
