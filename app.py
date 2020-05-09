import click
import requests
from clint.textui import progress
import threading
import os
from zipfile import ZipFile
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import PIL
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.nn import conv2d, avg_pool2d, dropout, relu, softmax_cross_entropy_with_logits_v2
from tensorflow.layers import flatten


classes = {
    "00" : "speed limit 20 (prohibitory)",
    "01" : "speed limit 30 (prohibitory)",
    "02" : "speed limit 50 (prohibitory)",
    "03" : "speed limit 60 (prohibitory)",
    "04" : "speed limit 70 (prohibitory)",
    "05" : "speed limit 80 (prohibitory)",
    "06" : "restriction ends 80 (other)",
    "07" : "speed limit 100 (prohibitory)",
    "08" : "speed limit 120 (prohibitory)",
    "09" : "no overtaking (prohibitory)",
    "10" : "no overtaking (trucks) (prohibitory)",
    "11" : "priority at next intersection (danger)",
    "12" : "priority road (other)",
    "13" : "give way (other)",
    "14" : "stop (other)",
    "15" : "no traffic both ways (prohibitory)",
    "16" : "no trucks (prohibitory)",
    "17" : "no entry (other)",
    "18" : "danger (danger)",
    "19" : "bend left (danger)",
    "20" : "bend right (danger)",
    "21" : "bend (danger)",
    "22" : "uneven road (danger)",
    "23" : "slippery road (danger)",
    "24" : "road narrows (danger)",
    "25" : "construction (danger)",
    "26" : "traffic signal (danger)",
    "27" : "pedestrian crossing (danger)",
    "28" : "school crossing (danger)",
    "29" : "cycles crossing (danger)",
    "30" : "snow (danger)",
    "31" : "animals (danger)",
    "32" : "restriction ends (other)",
    "33" : "go right (mandatory)",
    "34" : "go left (mandatory)",
    "35" : "go straight (mandatory)",
    "36" : "go right or straight (mandatory)",
    "37" : "go left or straight (mandatory)",
    "38" : "keep right (mandatory)",
    "39" : "keep left (mandatory)",
    "40" : "roundabout (mandatory)",
    "41" : "restriction ends (overtaking) (other)",
    "42" : "restriction ends (overtaking (trucks)) (other)",

}

IMG_SIZE = 32


def load_data(data_dir):
    data = []
    labels = []
    data_dirs = os.listdir(data_dir)
    for im_dir in data_dirs:
        for im in os.listdir(os.path.join(data_dir, im_dir)):
            image = Image.open(os.path.join(data_dir, im_dir, im))
            data.append(np.array(image.resize((IMG_SIZE, IMG_SIZE), PIL.Image.ANTIALIAS)))
            labels.append(im_dir)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def load_user_data(data_dir):
    data = []
    data_dirs = os.listdir(data_dir)
    for im in data_dirs:
        image = Image.open(os.path.join(data_dir, im))
        data.append(np.array(image.resize((IMG_SIZE, IMG_SIZE), PIL.Image.ANTIALIAS)))

    data = np.array(data)
    return data


def load_sk_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model

def train_logistic_model(data_dir):
    train_data, y_train = load_data(data_dir)
    X = train_data.reshape(train_data.shape[0], -1)
    X_train = X / 255
    model = LogisticRegression(verbose=2)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)

    click.echo(f"Accuracy on training data: {train_score * 100}%")

    save_dir = "models/model1/saved"
    with open(os.path.join(save_dir, "model1.pkl"), "wb") as f:
        pickle.dump(model, f)

def test_logistic_model(data_dir):
    test_data, y_test = load_data(data_dir)
    X = test_data.reshape(test_data.shape[0], -1)
    X_test = X / 255

    model = load_sk_model("models/model1/saved/model1.pkl")

    score = model.score(X_test, y_test)
    click.echo(f"Test accuracy: {score}")

def infer_logistic_model(data_dir):
    model = load_sk_model("models/model1/saved/model1.pkl")
    data = load_user_data(data_dir)
    X_user = data.reshape(data.shape[0], -1)
    X = X_user / 255

    preds = model.predict(X)

    for i, pred in enumerate(preds):
        fig = plt.figure()
        plt.imshow(data[i])
        plt.title(f"{classes[pred]}")

    plt.show()

def train_tensorflow_model(data_dir):
    labels = np.array(list(classes.keys())).reshape(len(classes), 1)
    learning_rate = 0.01
    num_iterations = 1000

    X_train, y_train = load_data(data_dir)
    X_train = X_train / 255
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels)
    y_train = encoder.transform(np.array(y_train).reshape(len(y_train), 1))

    X = tf.placeholder(tf.float32, [None, 3 * IMG_SIZE ** 2], name="X")
    b = tf.Variable(tf.zeros([len(labels)]))
    W = tf.Variable(tf.random_normal([3 * IMG_SIZE ** 2, len(labels)]))

    y = tf.add(tf.matmul(X, W), b, name="y")
    y_true = tf.placeholder(tf.float32, [None, len(labels)], name="y_true")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_iterations):
            
            #Training
            batch_x = X_train.reshape(X_train.shape[0], -1)        
            sess.run(train, feed_dict={X: batch_x, y_true: y_train})

            if i % 100 == 0:
                correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                loss_val = sess.run(loss, feed_dict={X: batch_x, y_true: y_train})
                acc_val = sess.run(acc, feed_dict={X: batch_x, y_true: y_train})
                click.echo(f"Iteration: {i} => Accuracy: {acc_val}, Loss: {loss_val}")
            
        saver = tf.train.Saver()
        save_path = "models/model2/saved/model2"
        saver.save(sess, save_path)

def test_tensorflow_model(data_dir):
    img_size = 50
    labels = np.array(list(classes.keys())).reshape(len(classes), 1)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels)
    X_test, y_test = load_data(data_dir)
    X_test = X_test / 255
    y_test = encoder.transform(np.array(y_test).reshape(len(y_test), 1))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("models/model2/saved/model2.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models/model2/saved/"))

        graph = tf.get_default_graph()

        y_true = graph.get_tensor_by_name("y_true:0")
        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")
        preds = tf.nn.softmax(y)
        correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        click.echo(f"Accuracy on test set: {acc.eval({X: X_test.reshape(X_test.shape[0], -1), y_true: y_test})}")

def infer_tensorflow_model(data_dir):
    labels = np.array(list(classes.keys())).reshape(len(classes), 1)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels)
    data = load_user_data(data_dir)
    data = data / 255

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("models/model2/saved/model2.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models/model2/saved/"))

        graph = tf.get_default_graph()
        y = graph.get_tensor_by_name("y:0")
        X = graph.get_tensor_by_name("X:0")
        softmax_results = sess.run(tf.nn.softmax(y), feed_dict={X: data.reshape(data.shape[0], -1)})
        preds = sess.run(tf.argmax(softmax_results, 1))

    for i, pred in enumerate(preds):
        orig_pred = f"{pred}".zfill(2)
        fig = plt.figure()
        plt.imshow(data[i])
        plt.title(f"{classes[orig_pred]}")
    
    plt.show()

def conv_layer(x, num_inputs, num_filters, filter_shape, name):
    w = tf.Variable(tf.truncated_normal(shape=(filter_shape, filter_shape, num_inputs, num_filters), mean=0, stddev=0.1))
    b = tf.Variable(tf.zeros(num_filters))

    return tf.add(conv2d(x, w, strides=1, padding='VALID'), b, name=name)

def fully_connected(x, num_inputs,num_outputs, name):
    w = tf.Variable(tf.truncated_normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.1))
    b = tf.Variable(tf.zeros(num_outputs))

    return tf.add(tf.matmul(x, w), b, name=name)

def train_lenet_model(data_dir):
    labels = np.array(list(classes.keys())).reshape(len(classes), 1)
    learning_rate = 0.01
    X_train, y_train = load_data("images/train")
    X_train =  X_train / 255
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels)
    y_train = encoder.transform(np.array(y_train).reshape(len(y_train), 1))

    X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="X")
    y_true = tf.placeholder(tf.float32, shape=(None, len(labels)), name="y_true")

    C1 = conv_layer(X, 3, 6, 5, "C1")
    S2 = avg_pool2d(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="S2")
    S2_drop = dropout(S2, 0.8)
    C3 = conv_layer(S2_drop, 6 ,16, 5, name="C3")
    S4 = avg_pool2d(C3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="S4")
    S4_flatten = flatten(S4)
    FC1 = fully_connected(S4_flatten, 400, 120, "FC1")
    FC1 = relu(FC1)
    FC2 = fully_connected(FC1, 120, 84, "FC2")
    FC2 = relu(FC2)
    y_pred = fully_connected(FC2, 84, len(labels), "y")

    loss = tf.reduce_mean(softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    result = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(result, tf.float32))

    epochs = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Starting training of LeNet model", end="\n")
        for epoch in range(epochs):
            opt, accuracy, cur_loss = sess.run([train, acc, loss], feed_dict={X: X_train, y_true: y_train})
            if epoch % 10 == 0:
                print(f"EPOCH: {epoch}")
                print(f"Accuracy: {accuracy}")
                print(f"Loss: {cur_loss}", end="\n")
                print("====================================")
            
        saver = tf.train.Saver()
        save_path = "models/model3/saved/model3"
        saver.save(sess, save_path) 

def test_lenet_model(data_dir):

    labels = np.array(list(classes.keys())).reshape(len(classes), 1)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels)
    X_test, y_test = load_data(data_dir)
    X_test = X_test / 255
    y_test = encoder.transform(np.array(y_test).reshape(len(y_test), 1))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("models/model3/saved/model3.meta")
        saver.restore(sess, tf.train.latest_checkpoint("models/model3/saved/"))

        graph = tf.get_default_graph()

        y_true = graph.get_tensor_by_name("y_true:0")
        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")
        preds = tf.nn.softmax(y)
        correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        click.echo(f"Accuracy on test set: {acc.eval({X: X_test, y_true: y_test})}")

@click.group()
def cli():
    pass


@cli.command()
@click.option("-m", help="Model to train", required=True)
@click.option("-d", help="Model to train", required=True)
def train(m, d):
    click.echo(f"Training {m} model")
    if m == "logistic":
        train_logistic_model(d)
    elif m == "tensorflow":
        train_tensorflow_model(d)
    elif m == "lenet":
        train_lenet_model(d)

@cli.command()
@click.option("-m", default="logistic", help="Model to train")
@click.option("-d", help="Model to train", required=True)
def test(m, d):
    click.echo(f"Testing {m} model")
    if m == "logistic":
        test_logistic_model(d)
    elif m == "tensorflow":
        test_tensorflow_model(d)
    elif m == "lenet":
        test_lenet_model(d)


@cli.command()
@click.option("-m", default="logistic", help="Model to train")
@click.option("-d", help="Model to train", required=True)
def infer(m, d):
    click.echo(f"Predicting {m} model")
    if m == "logistic":
        infer_logistic_model(d)
    elif m == "tensorflow":
        infer_tensorflow_model(d)

@cli.command()
def download():
    data_url = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"

    if not os.path.exists("image"):
        os.mkdir("image")

    if not os.path.exists("image/data.zip"):
        r = requests.get(data_url, stream=True)

        with open("images/data.zip", "wb") as f:
            total_length = int(r.headers.get("content-length"))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()

    with ZipFile("images/data.zip", "r") as zipObj:
        zipObj.extractall("images/")

    data_dir = "images/FullIJCNN2013"

    data = []
    labels = []


    for f in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, f)):
            for im in os.listdir(os.path.join(data_dir, f)):
                data.append(os.path.join(data_dir, f, im))
                labels.append(f)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    for i in range(len(X_train)):
        dest_dir = f"images/train/{y_train[i]}"
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        shutil.move(X_train[i], dest_dir)

    for i in range(len(X_test)):
        dest_dir = f"images/test/{y_test[i]}"
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        shutil.move(X_test[i], dest_dir)
    shutil.rmtree(data_dir)

if __name__ == "__main__":
    cli()