import numpy as np
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags
import os
FLAGS = flags.FLAGS
flags.DEFINE_float('update_lr', 10, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_bool('init_with_training_data', False, 'Whether init k and v with training data.')
flags.DEFINE_integer('num_slots', 100, 'Number of slots in memory.')
flags.DEFINE_bool('vanilla', False, 'if true, run vanilla model.')
flags.DEFINE_integer('num_layers', 0, 'number of fully connected hidden layers.')
flags.DEFINE_integer('num_neurons', 2, 'number of neurons at each layer.')
def make_one_hot(y, num_classes=2):
    one_hot = np.zeros((y.shape[0], num_classes))
    for i in range(y.shape[0]):
        one_hot[i, int(y[i])] = 1
    return one_hot
def twospirals(n_points, noise=1, seed=1):
    """
     Returns the two spirals dataset.
    """
    np.random.seed(seed)
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))

def get_data(n_points, noise, seed):
    X, y = twospirals(n_points,noise,seed)
    data = np.concatenate([X,np.expand_dims(y,1)], axis=1)
    np.random.shuffle(data)
    X = data[:,:2]
    y = data[:,2]
    X = (X - X.mean()) / X.std()
    plt.title('training set')
    plt.plot(X[y==0,0], X[y==0,1], '.', label='class 1')
    plt.plot(X[y==1,0], X[y==1,1], '.', label='class 2')
    plt.legend()
    #plt.show()
    y = np.expand_dims(y,axis=1)
    return X,y

X_train, y_train = get_data(800,1,1)
X_val, y_val = get_data(800,1,2)
if FLAGS.init_with_training_data:
    k_init, v_init = get_data(int(FLAGS.num_slots / 2), 1, 4)

def cos_similarity(query,key, alpha):
    # query: 1* num_features
    # key: num_slots * num_features 
    query_norm = tf.nn.l2_normalize(query, axis=1)
    key_norm = tf.nn.l2_normalize(key, axis=1)
    cos_sim = tf.reduce_sum(alpha*tf.multiply(query_norm,key_norm), axis=1)
    cos_sim = tf.nn.softmax(cos_sim)
    return cos_sim

def euclidian_similarity(query, key, alpha):
    euclidian_dist = alpha*tf.reduce_sum(tf.multiply((query-key),(query-key)), axis=1)
    return tf.nn.softmax(-euclidian_dist)
def weighted_sigmoid_binary_cross_entropy_loss(pred, label, weights):
    # pred: num_slots * num_classes
    # label: num_slots * num_classes
    # weights: num_slots
    y_true = label
    y_hat = tf.nn.sigmoid(pred)
    y_cross = tf.squeeze(-y_true*tf.math.log(y_hat + 1e-8)-(1-y_true)*tf.math.log(1-y_hat+1e-8))
    result = weights * y_cross
    return result

class MODEL:
    def __init__(self, num_slots=100, update_steps=FLAGS.num_updates, num_fc_layers=FLAGS.num_layers, num_neurons=FLAGS.num_neurons):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.num_slots = num_slots
        self.update_steps = update_steps
        self.num_fc_layers = num_fc_layers
        self.num_neurons = num_neurons
        self.update_lr = FLAGS.update_lr
    def construct_vanilla_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_varialbes()
                weights = self.weights
                memo_weights = self.memo_weights
            else:
                self.weights = weights = self.construct_weights()
                self.memo_weights = memo_weights = self.construct_memory()
            out = self.forward(self.x, weights)
            self.pred = (out>0.5)
            vanilla_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.y))

        
        self.vanilla_loss = vanilla_loss
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.gvs = gvs = optimizer.compute_gradients(self.vanilla_loss)
        self.train_op = optimizer.apply_gradients(gvs)
        
    def construct_meta_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_varialbes()
                weights = self.weights
                memo_weights = self.memo_weights
            else:
                self.weights = weights = self.construct_weights()
                self.memo_weights = memo_weights = self.construct_memory()
            def task_metalearn(inp):
                x, y = inp
                sim = euclidian_similarity(x, memo_weights['k'], memo_weights['alpha'])
                max_idx = tf.argmax(sim)
                task_outputa = self.forward(memo_weights['k'], weights)
                task_lossa = weighted_sigmoid_binary_cross_entropy_loss(task_outputa, memo_weights['v'], sim)
                grads = tf.gradients(task_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                for i in range(self.update_steps - 1):
                    task_outputa = self.forward(memo_weights['k'], fast_weights)
                    task_lossa = weighted_sigmoid_binary_cross_entropy_loss(task_outputa, memo_weights['v'], sim)
                    grads = tf.gradients(task_lossa, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                outputb = self.forward(x, fast_weights)
                lossb = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputb, labels=y)
                return [outputb, lossb, sim, weights['b0'], fast_weights['b0'], max_idx, gradients['b0']]
            out_dtype = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.float32]
            result = tf.map_fn(task_metalearn, elems=(tf.expand_dims(self.x, axis=1),tf.expand_dims(self.y, axis=1)), dtype=out_dtype, parallel_iterations=32)
            self.outputb, lossb, self.sim, self.b_before, self.b_after, self.max_idx, self.gradients = result
            self.pred = (self.outputb>0.5)
            self.meta_loss = tf.reduce_mean(lossb)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.minimize(self.meta_loss)
            
            
            
    def forward(self, inp, weights):
        x = inp
        for i in range(self.num_fc_layers):
            x = tf.math.tanh(tf.matmul(x, weights['w'+str(i)]) + weights['b'+str(i)])
        
        x = tf.matmul(x, weights['w' + str(self.num_fc_layers)]) + weights['b' + str(self.num_fc_layers)]

        return x
    def construct_memory(self):
        memo_weights = {}
        dtype = tf.float32
        uniform_init = tf.initializers.random_uniform(minval=-2, maxval=2)
        normal_init = tf.initializers.truncated_normal(dtype = dtype, stddev=0.1)
        if FLAGS.init_with_training_data:
            memo_weights['v'] = tf.get_variable('v', initializer = v_init.astype('float32'), dtype=dtype, trainable=False)
            memo_weights['k'] = tf.get_variable('k', initializer = k_init.astype('float32'), dtype=dtype)
        else:
            memo_weights['v'] = tf.nn.sigmoid(tf.get_variable('v', shape=[self.num_slots, 1], initializer=tf.initializers.random_uniform(minval=-1000, maxval=1000), dtype=dtype, trainable=False))
            memo_weights['k'] = tf.get_variable('k', initializer=uniform_init, shape=[self.num_slots, 2], dtype=dtype)
        memo_weights['m'] = tf.get_variable('m', shape=[self.num_slots, 2], initializer=tf.initializers.truncated_normal(dtype = dtype, mean=1.6, stddev=0.4), dtype=dtype)
        memo_weights['alpha'] = tf.get_variable('alpha', initializer=tf.constant(5.0))
        return memo_weights
    def construct_weights(self):
        weights = {}
        dtype = tf.float32
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        with tf.variable_scope('fc', reuse=None):
            for i in range(self.num_fc_layers):
                if i == 0:
                    weights['w'+str(i)] = tf.get_variable('w'+str(i), [2, self.num_neurons], initializer=fc_initializer)
                    weights['b'+str(i)] = tf.get_variable('b'+str(i), [self.num_neurons], initializer=tf.zeros_initializer())
                else:
                    weights['w'+str(i)] = tf.get_variable('w'+str(i), [self.num_neurons, self.num_neurons], initializer=fc_initializer)
                    weights['b'+str(i)] = tf.get_variable('b'+str(i), [self.num_neurons], initializer=tf.zeros_initializer())
            weights['w'+str(self.num_fc_layers)] = tf.get_variable('w'+str(self.num_fc_layers), [self.num_neurons, 1], initializer=fc_initializer)
            weights['b'+str(self.num_fc_layers)] = tf.get_variable('b'+str(self.num_fc_layers), [1,1], initializer=tf.zeros_initializer())
            return weights
         

m = MODEL(num_slots=FLAGS.num_slots)
if FLAGS.vanilla:
    m.construct_vanilla_model()
else:
    m.construct_meta_model()

num_steps = 1000000
batch_size = 32
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()



if FLAGS.vanilla:
    if not os.path.exists('./vanilla'):
        os.makedirs('./vanilla')
    for itr in range(num_steps):
        feed_dict = {m.batch_size:32, m.x: X_train[itr*batch_size % 1600:(itr+1)*batch_size %1600,:], m.y: y_train[itr*batch_size%1600:(itr+1)*batch_size%1600,:]}
        input_tensors = [m.vanilla_loss, m.train_op]
        loss, _ = sess.run(input_tensors, feed_dict=feed_dict)
        if itr % 10 == 0:
            print(itr,loss)

            #plt.show()
            val_loss_total = 0
            count = 0
            preds = []
            for i in range(int(1600/32)):
                feed_dict = {m.batch_size:32, m.x: X_val[i*32:(i+1)*32,:], m.y: y_val[i*32:(i+1)*32]}
                val_loss, pred = sess.run([m.vanilla_loss, m.pred], feed_dict=feed_dict)
                count += 1
                preds.append(pred)
                val_loss_total += val_loss
            #print('b before')
            #print(b_before)
            #print('b_after')
            #print(b_after)
            print('validation loss:' + str(val_loss))
            #print('max idx')
            #print(max_idx)
            #print('gradients')
            #print(gradients)
        
            pred = np.concatenate(preds, axis=0)
            pred = np.squeeze(pred)
            fig1 = plt.gcf()
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.title('pred')
            plt.plot(X_val[pred==0,0], X_val[pred==0,1], 'bo', label='class 1')
            plt.plot(X_val[pred==1,0], X_val[pred==1,1], 'ro', label='class 2')
            plt.legend()
            plt.close()
            #plt.show()
            
            fig1.savefig('./vanilla/'+str(itr)+'.png')
else:
    if not os.path.exists('./'+str(FLAGS.update_lr)+'_'+str(FLAGS.num_updates)):
        os.makedirs('./'+str(FLAGS.update_lr)+'_'+str(FLAGS.num_updates))
    
    for itr in range(num_steps):
        feed_dict = {m.batch_size:32, m.x: X_train[itr*batch_size % 1600:(itr+1)*batch_size %1600,:], m.y: y_train[itr*batch_size%1600:(itr+1)*batch_size%1600,:]}
        input_tensors = [m.meta_loss, m.train_op]
        loss, _ = sess.run(input_tensors, feed_dict=feed_dict)
        if itr % 10 == 0:
            print(itr,loss)

            #plt.show()
            val_loss_total = 0
            count = 0
            preds = []
            for i in range(int(1600/32)):
                feed_dict = {m.batch_size:32, m.x: X_val[i*32:(i+1)*32,:], m.y: y_val[i*32:(i+1)*32]}
                val_loss, pred, key, v = sess.run([m.meta_loss, m.pred, m.memo_weights['k'], m.memo_weights['v']], feed_dict=feed_dict)
                count += 1
                preds.append(pred)
                val_loss_total += val_loss
            #print('b before')
            #print(b_before)
            #print('b_after')
            #print(b_after)g
            print('validation loss:' + str(val_loss))
            #print('max idx')
            #print(max_idx)
            #print('gradients')
            #print(gradients)
        
            pred = np.concatenate(preds, axis=0)
            pred = np.squeeze(pred)
            fig1 = plt.gcf()
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.title('pred')
            plt.plot(X_val[pred==0,0], X_val[pred==0,1], 'bo', label='class 1')
            plt.plot(X_val[pred==1,0], X_val[pred==1,1], 'ro', label='class 2')
            plt.plot(key[np.squeeze(v)<=0.5,0],key[np.squeeze(v)<=0.5,1],'y+', label='+keys')
            plt.plot(key[np.squeeze(v)>0.5,0],key[np.squeeze(v)>0.5,1],'yo', label='-keys')
            plt.legend()
            plt.close()
            #plt.show()
            fig1.savefig('./'+str(FLAGS.update_lr)+'_'+str(FLAGS.num_updates)+'/'+str(itr)+'.png')
