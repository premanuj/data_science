from keras.datasets import imdb

#Loading the IMBD dataset

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0:3])
print(train_labels[0])
print(max([max(sequence) for sequence in train_data]))