import numpy as np
import tensorflow as tf

train=np.load('train.npz')

val=np.load('valid.npz')
train_x1=train['ques1']
train_x2=train['ques2']
train_y=train['label']
max_length = train_x2.shape[1]
print max_length
val_x1=val['ques1']
val_x2=val['ques2']
val_y=val['label']
vec=np.load('embed.npz')
vec_x=vec['embed']
vec_x_ =tf.constant(vec_x,dtype=tf.float32)


print vec_x_
config = {"model": "RNN",
         "num_layer": 1,
         "dim": 128,
         "epoch": 10,
          "lr":1e-2,
          "seq_len": max_length,
          "optimizer": "sgd",
         }

class RNN:
    def __init__(self, config):
        self.model = config["model"]
        self.num_layer = config["num_layer"]
        self.dim = config["dim"]
        self.optimizer = config["optimizer"]
        self.lr = config["lr"]
        self.seq_len = config["seq_len"]
        self.epoch = config["epoch"]

    def length(self, sequences):
        """
        Return sequence length for all the inputs

        Parameters
        ----------
            sequences: Tensor shape=[batch_size, sequence_length, input_dimension]
                sequence of inputs

        Returns
        -------
            length: int Tensor shape=[None]
                length of each sequence in the batch
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequences), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def last_relevant(self, output, length):
        """
        Return last relevant output state for each sequence

        Parameters
        ----------
            output: Tensor shape=[batch_size, sequence_length, output_dimension]
                output tensor from tf.nn.dynamic_rnn

            length: int Tensor shape=[batch_size]
                length of every sequence in the batch

        Returns
        -------
            relevant: Tensor shape=[batch_size, output_dimension]
                last relevant vector for each sequence, i.e. vector at length index for each sequence in output
        """
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def fit(self, train_x1, train_x2, val_x1, val_x2, train_y, val_y):

        input_dim = vec_x.shape[1]
        output_class = 2

        x1 = tf.placeholder(shape=[None, self.seq_len], dtype=tf.int32)
        x2 = tf.placeholder(shape=[None, self.seq_len], dtype=tf.int32)
        y = tf.placeholder(shape=[None], dtype=tf.float32)

        optimize_op, loss, preds, logits ,preds1,preds2, summary= self.train(x1, x2, y)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # create file writer directory to store summary and events
            train_writer = tf.summary.FileWriter('./train', sess.graph)
            valid_writer = tf.summary.FileWriter('./valid')
            # train_summary=""
            # valid_summary=""

            for i in range(10):

                end = len(train_x1)
                count_loop = int(len(train_x2)/100)
                start=1
                for j in range(count_loop):
                    #print start*j
                    train1_batch,train2_batch,target_batch=self.get_batch(start*j,train_x1,train_x2,train_y)
                    
                                                         
                    loss1, _,out_val, train_summary = sess.run([loss, optimize_op,preds, summary], feed_dict={x1: train1_batch, x2: train2_batch, y: target_batch})

                    # logg training dataset
                train_writer.add_summary(train_summary, i)

                print ("train loss: ", loss1)

                #if i % 10 == 0:
                loss_val,out_val, valid_summary = sess.run([loss, preds, summary], feed_dict={x1: val_x1, x2: val_x2, y: val_y})
                # logg validation dataset
                valid_writer.add_summary(valid_summary, i)
                print ("validation loss: ", loss_val)

    def get_batch(self,start,train_x1,train_x2,y):
        return train_x1[start:start + 100], train_x2[start:start + 100], y[start:start + 100]

    def train(self, x1, x2, y):

        preds, logits ,preds1,preds2 = self._model(x1, x2)
        print preds.get_shape()
        print logits.get_shape()
        y= tf.reshape(y,[-1,1])
        #y_embed = tf.nn.embedding_lookup(vec_y,y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
     #   loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(logits, y))))
        optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

        # tensorboard sumary
        tf.summary.scalar('loss', loss)

        # merge all summary as an Op
        merged = tf.summary.merge_all()

        return optimizer, loss,preds,logits,preds1,preds2, merged

    def _model(self, x1, x2):
        x1 = tf.cast(x1, dtype=tf.int32)
        x2 = tf.cast(x2, dtype=tf.int32)

        with tf.name_scope(name="network"):
            cells = []
            for i in range(self.num_layer):
                with tf.variable_scope("cell{}".format(i), initializer=tf.contrib.layers.xavier_initializer()):
                    cell = tf.contrib.rnn.BasicRNNCell(self.dim)
                cells.append(cell)

            stacked_cell = tf.contrib.rnn.MultiRNNCell(cells)
            x1_embed = tf.nn.embedding_lookup(vec_x_,x1)

            x2_embed = tf.nn.embedding_lookup(vec_x_,x2)
            #x1 = tf.transpose(x1, [1, 0, 2])
            #x2 = tf.transpose(x2, [1, 0, 2])
            x1_embed_ = tf.cast(x1_embed, dtype=tf.float32)
            x2_embed_ = tf.cast(x2_embed, dtype=tf.float32)

            # significant sequence length for every input sequence without paddings
            x1_seq_len = self.length(x1_embed_)
            x2_seq_len = self.length(x2_embed_)

            with tf.variable_scope("siamese") as scope:
                output1, state1 = tf.nn.dynamic_rnn(cell=stacked_cell, inputs=x1_embed_, sequence_length=x1_seq_len,
                                                dtype=tf.float32)
                scope.reuse_variables()
                output2, state2 = tf.nn.dynamic_rnn(cell=stacked_cell, inputs=x2_embed_, sequence_length=x2_seq_len,
                                                    dtype=tf.float32)


            # last relevant output from unrolled RNN graph
            # output post `seq_len` index are zeros for every input sequence
            preds1 = self.last_relevant(output1, x1_seq_len)
            preds2 = self.last_relevant(output2, x2_seq_len)


            # preds1 = state1[-1]
           # preds1 = preds1[-1]

            # preds2 = state2[-1]

           # preds2 = preds2[-1]
            preds = tf.abs(preds1 - preds2)
           # vect_sum = tf.reduce_sum(preds, axis=1)
            with tf.variable_scope("fully_connected"):
                preds = tf.contrib.layers.fully_connected(
                    inputs=preds,
                    num_outputs=1)

                # output activation applied seperately for objective function
                logits = preds

            #logits_ = tf.nn.sigmoid(vect_sum)
           # logits = 1-logits
            #logits = tf.reshape(logits_,[-1,1])

        return preds, logits,preds1,preds2

ob = RNN(config)
ob.fit(train_x1,train_x2,val_x1,val_x2,train_y,val_y)