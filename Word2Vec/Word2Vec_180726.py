import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf

filename = 'text8.zip'
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename=filename)
print(len(words))
#print(words)
vocabulary_size = 50000
#构造四大类基础数据   data, count, dictionary, reverse_dictionary
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)
#print(data)
#print(count)
#print(dictionary)
#print(reverse_dictionary)
del words
#print(count[:5])
#print(data[:10])
#print(reverse_dictionary[1])
#print([reverse_dictionary[i] for i in data[:10]])
data_index = 0
#某个单词   与之相关的单词  的编号（编号为出现次数排行）
def generate_batch(batch_size, num_skips, skip_window):
    #skip_window 最远链接距离
    #num_skips 每个单词最少生成多少样本
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype = np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1#创建样本是需要用到的单词数量，包括单词本身
    buffer = collections.deque(maxlen=span)#双向队列
    for _ in range(span):
        buffer.append(data[data_index])#data中单词
        data_index = (data_index+1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips +j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1)%len(data)
    return batch, labels

#batch, labels = generate_batch(8, 2,1)
#for i in range(8):
#    print(batch[i],reverse_dictionary[batch[i]],'---->',labels[i, 0],reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    #with tf.device('/cpu:0'):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize((loss))
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embedding, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()
    num_steps = 100001
    with tf.Session(graph=graph) as session:
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
            _, loss_val=session.run([optimizer, loss],feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Average loss at step',step,':',average_loss)
                average_loss = 0

            if step % 10000 ==0:
                sim = similarity.eval()

                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log_str = 'Nearest to %s:' % valid_word

                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s,"%(log_str, close_word)
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()