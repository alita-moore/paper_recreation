from DCT_DWT_SVM import load_data, test_svm


data = load_data.load_data()
print('data got')

n = 1000

# sample_EMNIST = [data[0][0:n], data[1][0:n]]
# label_EMNIST = [data[2][0:n], data[3][0:n]]
# test(sample_EMNIST, label_EMNIST)

features = 196
sample_MNIST = [data[4][0:n], data[5][0:n]]
label_MNIST = [data[6][0:n], data[7][0:n]]
test_svm.test_svm(sample_MNIST, label_MNIST, features)