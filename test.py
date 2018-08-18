import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv

def get_features_column_count():
	return 3;
def get_targets_column_count():
	return 1;

def get_dataset():
        """
                Method used to generate the dataset
        """
	features = []
	targets = []

        f = open("data.csv", "r")
        reader = csv.reader(f, delimiter=',')
        for row in reader:
                features.append(row)
	print "features\n", features, "\n"

        f = open("result.csv", "r")
        reader = csv.reader(f, delimiter=',')
        for row in reader:
                targets.append(row)
	print "targets\n", targets, "\n"

        return features, targets

if __name__ == '__main__':
        features, targets = get_dataset()
        sess = tf.Session()

        # Plot points
        #plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
        #plt.show()
	
	# tf_features = tf.placeholder(tf.float32, shape=[lentgh, Nb of data for row])
        tf_features = tf.placeholder(tf.float32, shape=[None, get_features_column_count()])
	# tf_targets = tf.placeholder(tf.float32, shape=[lentgh, Nb of result for row])
        tf_targets = tf.placeholder(tf.float32, shape=[None, get_targets_column_count()])

	# features Line
	flp = tf.Variable(tf.random_normal([get_features_column_count(), 1]))
	fls = tf.Variable(tf.random_normal([get_features_column_count(), 1]))
	# features prediction
	fp = tf.matmul(tf_features, flp) + fls
	# error prediction
	ep  = tf.reduce_mean(tf.square(fp - tf_targets))
	# optimiseur
	train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(ep)
	# accuracy (ratio juste)
	correct_prediction = tf.equal(tf.round(fp), tf_targets)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Learning :
        sess.run(tf.global_variables_initializer())
	

        sess.run(tf_features, feed_dict={
                tf_features: [features[0]],
                tf_targets: [targets[0]]
        })
        print "flp = ", sess.run(flp), "\n"
        print "fls = ", sess.run(fls), "\n"
        print "fp = ", sess.run(fp, feed_dict={
                tf_features: [features[0]],
                tf_targets: [targets[0]]
        }), "\n"
        print "ep = ", sess.run(ep, feed_dict={
                tf_features: [features[0]],
                tf_targets: [targets[0]]
        }), "\n"
        print "train_op = ", sess.run(train_op, feed_dict={
                tf_features: [features[0]],
                tf_targets: [targets[0]]
        }), "\n"
        print "correct_prediction = ", sess.run(correct_prediction, feed_dict={
                tf_features: [features[0]],
                tf_targets: [targets[0]]
        }), "\n"
        print "accuracy = ", sess.run(accuracy, feed_dict={
                tf_features: [features[0]],
                tf_targets: [targets[0]]
        }), "\n"

