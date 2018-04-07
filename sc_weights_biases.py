# 
# # coding: utf-8
# 
# # # Question Classification
# 
# # In[1]:
# 
# 
# import pandas as pd
# from gensim.models import Word2Vec
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# import numpy as np
# from libraries.pickleFunctions import writeObjToPickle,loadPickle
# 
# 
# # # Load Word2Vec Model
# # We trained a word2vec model on wikipedia corpus and have it saved.(code can be found in wikipedia_full_train.py file). We load this model and set up variables like vocabulary dict, embedding size, unknown word representation.
# 
# # In[2]:
# 
# 
# def get_word2vec_model():
#     model = Word2Vec.load("/home/rohit.katariya/rohit/sentence_classification/datadump/wiki_full_model.text.model")
#     return model
# 
# 
# # In[3]:
# 
# 
# w2v_model = get_word2vec_model()
# 
# 
# # In[4]:
# 
# 
# w2v_vocabulary=w2v_model.wv.vocab
# 
# 
# # In[5]:
# 
# 
# size_embeddings_vector = w2v_model.trainables.layer1_size
# 
# 
# # In[6]:
# 
# 
# unk_word_rep = [0.0]* size_embeddings_vector
# 
# 
# # # Read Dataset
# # 
# # We read the dataset of question pairs.
# 
# # In[7]:
# 
# 
# data_filename="datadump/train.csv"
# 
# 
# # In[8]:
# 
# 
# data_df = pd.read_csv(data_filename)
# 
# 
# # # Convert Sentences to Embeddings
# # We write a function to be used later to convert a sentence to its corresponding word2vec representation, padding with *unknown* word representation in front in case it is smaller than the max size of sentence fixed. We also truncate very large sentences(this was very small 0.02%). 
# 
# # In[9]:
# 
# 
# def apply_embedding_to_sentence(sentence,max_sentence_len):
#     sentence= str(sentence)
#     sentence_words = sentence.split()
#     sentence_embeddings = []
#     for sentence_word in sentence_words:
#         if sentence_word in w2v_vocabulary:
#             sentence_emb = w2v_model.wv.__getitem__(sentence_word)
#             sentence_embeddings.append( np.asarray(sentence_emb) )
#         else:
#             sentence_embeddings.append(unk_word_rep)
#     sentence_embeddings = sentence_embeddings[:max_sentence_len]
#     sentence_embeddings = [unk_word_rep]*(max_sentence_len - len(sentence_embeddings)) + sentence_embeddings
#     return  np.asarray( sentence_embeddings )
# 
# 
# # # Fix maximum length that a sentence can take up
# # 
# # We find the maximum length a sentence a sentence can take and fix it.
# 
# # In[10]:
# 
# 
# def get_sentence_len(sen):
#     sen= str(sen)
#     return len(sen.split())
# 
# 
# # In[11]:
# 
# 
# len_each_1 = data_df["question1"].apply(get_sentence_len)
# len_each_2 = data_df["question2"].apply(get_sentence_len)
# len_each = pd.concat([len_each_1,len_each_2])
# 
# 
# # In[12]:
# 
# 
# len_each.describe()
# 
# 
# # In[13]:
# 
# 
# max_sentence_len = int(len_each.quantile(0.998))
# 
# 
# # # Apply embedding transformations to sentences
# 
# # In[14]:
# 
# 
# data_df["question1_emb"]=data_df.question1.apply(apply_embedding_to_sentence,max_sentence_len=max_sentence_len)
# data_df["question2_emb"]=data_df.question2.apply(apply_embedding_to_sentence,max_sentence_len=max_sentence_len)
# 
# 
# # # Prepare training, testing datasets
# 
# # In[15]:
# 
# 
# data_df_processed_X = data_df[['question1_emb','question2_emb']]
# data_df_processed_Y = data_df[['is_duplicate']]
# 
# 
# # In[16]:
# 
# 
# x_train, x_test, y_train, y_test = train_test_split(data_df_processed_X, data_df_processed_Y, test_size=5000*1.0/len(data_df_processed_X), random_state=1)
# 
# 
# # In[17]:
# 
# 
# print( "length of train test split:" ,[len(s) for s in [y_train, y_test] ])
# 
# 
# # In[32]:
# 
# 
# np.asarray(list(x_train.question1_emb)).shape
# 
# 
# # # Save pre-processed data to disc
# 
# # Save the training test set to disc so that we don't have to reload full word2vec model every time we want to load data.
# 
# # In[33]:
# 
# 
# x_train.to_pickle("datadump/preprocessed_data/xtrain.pkl")
# x_test.to_pickle("datadump/preprocessed_data/x_test.pkl")
# y_train.to_pickle("datadump/preprocessed_data/y_train.pkl")
# y_test.to_pickle("datadump/preprocessed_data/y_test.pkl")
# 
# 
# # In[34]:
# 
# 
# # import pickle
# # def writeObjToPickle(dict_,filename):
# #     """
# #         writes an object to pickle file named filename.
# #     """
# #     file_dict = open(filename+'.pickle', 'wb')
# #     pickle.dump(dict_, file_dict)
# #     file_dict.close()
# 
# 
# # In[35]:
# 
# 
# # max_sentence_len
# # from libraries.pickleFunctions import writeObjToPickle,loadPickle
# writeObjToPickle( max_sentence_len,"datadump/preprocessed_data/max_sentence_len")
# writeObjToPickle( size_embeddings_vector,"datadump/preprocessed_data/size_embeddings_vector")
# 
# 
# # In[36]:
# 
# 
# print("Done writing preprocessed data.")


# # Checkpoint
# We saved the sentence data and now we can start loading these from here. We load the train,test data.

# In[1]:


import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from libraries.pickleFunctions import writeObjToPickle,loadPickle
import random 
import csv

def writeDictListToCSV(dicList, filename, delimiter=None, filemode='w', column_names_set=None):
    """
        Writes a list of dictionaries to a csv file with each dictionary as a row
    """   
    if not column_names_set:
        column_names_set = set([])
        for row in dicList:
            for k in row.keys():
                column_names_set.add(k)
    with open(filename + '.csv', filemode) as output_file:
        if not delimiter:
            dict_writer = csv.DictWriter(output_file, column_names_set)
        else:
            dict_writer = csv.DictWriter(output_file, column_names_set, delimiter=delimiter)
        dict_writer.writeheader()
        dict_writer.writerows(dicList)
# In[2]:


x_train = pd.read_pickle("datadump/preprocessed_data/    .pkl")
x_test = pd.read_pickle("datadump/preprocessed_data/x_test.pkl")
x_valid = pd.read_pickle("datadump/preprocessed_data/x_valid.pkl")
y_train = pd.read_pickle("datadump/preprocessed_data/y_train.pkl")
y_test = pd.read_pickle("datadump/preprocessed_data/y_test.pkl")
y_valid = pd.read_pickle("datadump/preprocessed_data/y_valid.pkl")


# In[3]:


def bin_transform(lables_sample):
    ohe_vec = np.zeros( (len(lables_sample),2), dtype=np.int)
    for x_ind in range(len(lables_sample)):
        ohe_vec[x_ind] = np.zeros(2, dtype=np.int)
        ohe_vec[x_ind][lables_sample[x_ind]] = 1
    ohe_vec_dict = {}
    for k in range(len(ohe_vec)):
        ohe_vec_dict[k] = np.asarray( ohe_vec[k])
    return ohe_vec_dict


# In[4]:


transform_dict = bin_transform([0,1])


# In[5]:


y_train_transformed = y_train.is_duplicate.apply(transform_dict.get)


# In[6]:


y_test_transformed = y_test.is_duplicate.apply(transform_dict.get)


# In[7]:


print( "length of train test split:" ,[len(s) for s in [y_train, y_test] ])


# In[8]:


max_sentence_len = loadPickle("datadump/preprocessed_data/max_sentence_len.pickle")
size_embeddings_vector = loadPickle("datadump/preprocessed_data/size_embeddings_vector.pickle")


# # Split data into batches
# We convert data into batches of batch_size sizes.

# In[9]:


display=1
batch_size=1000


# In[10]:


def split_into_batches(batch_size,df_X, df_Y):
    outout_batches = []    
    sample_size = len(df_Y)
    for start_i in range(0, len(df_X), batch_size):
        end_i = start_i + batch_size
        batch = [df_X[start_i:end_i], df_Y[start_i:end_i]]
        outout_batches.append(batch)
    return outout_batches


# In[11]:


# def split_into_batches(pd_df,batch_size):
#     return np.split(pd_df, len(pd_df)/(batch_size+1)) 


# In[38]:


training_batches = split_into_batches(batch_size,x_train, y_train.is_duplicate)
# testing_batches = split_into_batches(batch_size,x_test, y_train.is_duplicate)
# batch_x1_list, batch_x2_list, batch_y_list = np.split(x_train.question1_emb, len(x_train.question1_emb)/128) ,x_train.question2_emb, y_train.is_duplicate


# In[13]:


# training_batches[0][1]


# # Set hyperparameters

# In[14]:


n_inputs = size_embeddings_vector
n_steps = max_sentence_len 
n_hidden = 128 
n_classes = 2


# In[15]:


sentence_1 = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs],name="sentence_1")
sentence_2 = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs],name="sentence_2")
y = tf.placeholder(tf.int64, shape=[None],name = 'y')
keep_prob = tf.placeholder(tf.float32,name ="keep_prob")


# In[16]:


def reshape_sentence_tensor(sentence_tf,n_steps):
    # input: sentence_tf: tensor of shape [batch_size, n_steps, n_inputs]
    #output: sentence_tf_new: list of tensors with dimensions [batch_size, n_inputs]; len of this list n_steps
    # first we bring the second dimension n_steps in front
    transposed_tf = tf.transpose(sentence_tf, [1, 0, 2])   #[n_steps, batch_size, n_inputs]
    # we then split the tensor into n_step tensors
    splitted_tf_list = tf.split(transposed_tf,axis=0,num_or_size_splits=n_steps)  # list of tensors [1,batch_size, n_inputs]
    # now we will remove the first dimension from each tensor using squeez function
    squezed_tf_list = [tf.squeeze(splitted_tf, [0]) for splitted_tf in  splitted_tf_list] #list of tensors [batch_size, n_inputs]
    return squezed_tf_list


# In[17]:


sentence_1_list =reshape_sentence_tensor(sentence_1,n_steps)
sentence_2_list =reshape_sentence_tensor(sentence_2,n_steps)


# In[18]:


# []


# # Siamese Architecture
# Now that we have data in the format required by LSTM, we can start building our Siamese LSTM architecture.

# ## Bi-LSTM Cells
# First we create 2 LSTM cells of the bi-LSTM.

# In[19]:


forward_lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
backward_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)


# ## Dropout
# Insert dropout layer to avoid overfitting.

# In[20]:


forward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(forward_lstm_cell, output_keep_prob=keep_prob)
backward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(backward_lstm_cell, output_keep_prob=keep_prob)


# In[21]:


# state_fw = initial_state_fw = forward_lstm_cell.zero_state(batch_size, dtype=tf.float32)
# state_bw = initial_state_bw = backward_lstm_cell.zero_state(batch_size, dtype=tf.float32)


# In[22]:


# with tf.variable_scope('siamese_network') as scope:
#     with tf.name_scope('Bi_LSTM_1'):
#         _, last_state_fw1, last_state_bw1 = tf.nn.bidirectional_rnn(
#                                         lstm_fw_cell, lstm_bw_cell, x1_,
#                                         dtype=tf.float32)
#     with tf.name_scope('Bi_LSTM_2'):
#         scope.reuse_variables() # tied weights (reuse the weights from `Bi_LSTM_1` for `Bi_LSTM_2`)
#         _, last_state_fw2, last_state_bw2 = tf.nn.bidirectional_rnn(
#                                         lstm_fw_cell, lstm_bw_cell, x2_,
#                                         dtype=tf.float32)


# In[23]:


with tf.variable_scope("siamese_network") as scope:
    _, (last_state_fw1,last_state_bw1) = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_lstm_cell, 
                                                                         cell_bw=backward_lstm_cell, 
                                                                         inputs = sentence_1,
                                                                         dtype=tf.float32)
with tf.variable_scope(scope, reuse=True):
  _, (last_state_fw2, last_state_bw2) = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_lstm_cell, 
                                                                         cell_bw=backward_lstm_cell, 
                                                                         inputs = sentence_2,
                                                                         dtype=tf.float32)


# In[24]:


def get_rep_state(fw_state,bw_state):
    return tf.concat([fw_state[0],fw_state[1],bw_state[0],bw_state[1] ],1)


# In[25]:


s1_rep_out = get_rep_state(last_state_fw1,last_state_bw1)
s2_rep_out = get_rep_state(last_state_fw2,last_state_bw2)


# In[26]:


weights = tf.get_variable('weigths_out', shape=[4 * n_hidden, n_classes],
                initializer=tf.random_normal_initializer(stddev=1.0/float(n_hidden)))
biases = tf.get_variable('biases_out', shape=[n_classes])


# In[27]:


# logits =  tf.matmul(s1_rep_out - s2_rep_out, weights) + biases
logits = tf.nn.softmax(tf.matmul(s1_rep_out - s2_rep_out, weights) + biases )
# logits = tf.nn.relu(tf.add(tf.matmul(s1_rep_out - s2_rep_out, weights), biases)) 

# In[29]:


learning_rate = 0.001


# In[30]:


loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels= y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
pred_output = tf.argmax(logits, 1)
correct_pred = tf.equal(pred_output, y) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# precision = TP/TP+FP
# recall = TP/TP+FN
precision_output = tf.metrics.precision(labels = y,predictions = pred_output)
recall_output = tf.metrics.recall(labels = y,predictions = pred_output)


training_tuples = []
validation_tuples= []
testing_tuples = []
init = tf.global_variables_initializer()
max_iter = 140
n_validation = 5
# NUM_THREADS=30

# with tf.Session(config=tf.ConfigProto(sintra_op_parallelism_threads=NUM_THREADS )) as sess:
with tf.Session() as sess:
    sess.run(init)
    print('Network training begins.')
    for i in range(1, max_iter + 1):
        print("iteration:",i)
        batches_accuracy_list = []
        batches_loss_list =[]
        for batch_X, batch_Y in training_batches:
            batch_x1, batch_x2, batch_y = np.asarray(list(batch_X.question1_emb)),np.asarray(list(batch_X.question2_emb)), np.asarray(list(batch_Y)) 
            feed_dict = {sentence_1: batch_x1, sentence_2: batch_x2, y: batch_y, keep_prob: 0.5}
            _, loss_, accuracy_ = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
            batches_accuracy_list.append(accuracy_)
            batches_loss_list.append(loss_)
        training_tuples.append({
                                  "epoch":i,
                                  "loss":sum(batches_loss_list)/len(batches_loss_list),
                                  "accuracy":sum(batches_accuracy_list)/len(batches_accuracy_list)
                                  })
        if i % display == 0:
            print('step %i, training loss: %.5f, training accuracy: %.3f' % (i, loss_, accuracy_))
        # Testing the network on validation data
        if i % n_validation == 0:
            # Retrieving data from the test set
            #batch_no = random.randint(0, len(testing_batches))  
            #valid_batch = testing_batches[batch_no]
            valid_batch_X,valid_batch_Y= x_valid,y_valid.is_duplicate
            batch_x1, batch_x2, batch_y = np.asarray(list(valid_batch_X.question1_emb)),np.asarray(list(valid_batch_X.question2_emb)), np.asarray(list(valid_batch_Y))
            feed_dict = {sentence_1: batch_x1, sentence_2: batch_x2, y: batch_y, keep_prob: 1.0}
            #print( type(batch_y),batch_y.shape )
            loss_validation ,accuracy_validation= sess.run([ loss, accuracy], feed_dict=feed_dict)
            print('validation step %i, accuracy %.3f' % (i, accuracy_validation))
            validation_tuples.append({
                                        "epoch":i,
                                        "loss":loss_validation,
                                        "accuracy":accuracy_validation,
#                                         "precision_output":precision_output
#                                         "recall_output":sk.metrics.recall_score(batch_y, pred_output)
                                        })
    print('********************************')
    print('Training finished.')
    # testing the trained network on a large sample
    #test_batch = testing_batches[0]
    test_batch_X,test_batch_Y= x_test,y_test.is_duplicate
    batch_x1, batch_x2, batch_y = np.asarray(list(test_batch_X.question1_emb)),np.asarray(list(test_batch_X.question2_emb)), np.asarray(list(test_batch_Y))
    feed_dict = {sentence_1: batch_x1, sentence_2: batch_x2, y: batch_y, keep_prob: 1.0}
    #accuracy_test = sess.run(accuracy, feed_dict=feed_dict)
    loss_test ,accuracy_test,logits,y = sess.run([ loss, accuracy,logits,y], feed_dict=feed_dict)
    testing_tuples.append({
                        "loss":loss_test,
                        "accuracy":accuracy_test,
#                         "precision_output":sk.metrics.precision_score(y, logits),
#                         "recall_output":sk.metrics.precision_score(y, logits),
                        })
    print('********************************')
    print('Testing the network.')
    print('Network accuracy %.3f' % (accuracy_test))
    print('********************************')

writeDictListToCSV(validation_tuples,"datadump/validation_tuples")
writeDictListToCSV(training_tuples,"datadump/training_tuples")
writeDictListToCSV(testing_tuples,"datadump/testing_tuples")

# In[ ]:


# np.asarray(list(batch_Y)).shape


# In[ ]:


# np.asarray( list(batch_X.question1_emb)).shape


# In[ ]:


# init = tf. global_variables_initializer()
# max_iter = 4
# with tf.Session() as sess:
#     sess.run(init)
#     print('Network training begins.')
#     for i in range(1, max_iter + 1):
#         print("iteration:",i)
#         for batch_X, batch_Y in training_batches:
# #             sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
#             batch_x1, batch_x2, batch_y = np.asarray(list(batch_X.question1_emb)),np.asarray(list(batch_X.question2_emb)), np.asarray(list(batch_Y)) 
#             feed_dict = {sentence_1: batch_x1, sentence_2: batch_x2, y: batch_y, keep_prob: .9}
#             _, loss_ = sess.run([optimizer, loss ], feed_dict=feed_dict)
        
#         if i % display == 0:
#             print('step %i, training loss: %.5f, training accuracy: %.3f' % (i, loss_, accuracy_))
        

