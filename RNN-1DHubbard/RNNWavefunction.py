import tensorflow as tf
import numpy as np
import random


class RNNwavefunction(object):
    #selfはそのクラスの自身のこと
    def __init__(self,systemsize_x, systemsize_y,cell=tf.contrib.rnn.LSTMCell,activation=tf.nn.relu,units=[10],scope='RNNwavefunction',seed = 111):
        """
            systemsize_x:  int
                         number of sites for x-axis
            systemsize_y:  int
                         number of sites for y-axis         
            cell:        a tensorflow RNN cell
            activation:  activation function used for the RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:        pseudo-random number generator 
        """
        
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #x_size of the lattice
        self.Ny=systemsize_y #y_size of the lattice

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        #具体的にself.graphの中身の変数を初期化するためのコード
        with self.graph.as_default():
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                tf.set_random_seed(seed)  # tensorflow pseudo-random generator
                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                #RNNの深さと、それぞれの深さでの幅を設定出来るようになっているっぽい。units=[10]ならば１層の幅10で作られるが、units=[a, b, c,...]というように拡張できるようになっている。
                self.rnn=tf.nn.rnn_cell.MultiRNNCell([cell(units[n]) for n in range(len(units))])
                #最後に確率を取り出すためのNN
                self.dense = tf.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense', dtype = tf.float64) #Define the Fully-Connected layer followed by a Softmax


    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin
            ------------------------------------------------------------------------
            Returns:      
            samples:         tf.Tensor of shape (numsamples,systemsize_x*systemsize_y)
                             the samples in integer encoding
        """
        with self.graph.as_default(): #Call the default graph, used if not willing to create multiple graphs.
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                #生成確率に最初に入れる一番最初のsiteの状態
                b=np.zeros((numsamples,inputdim)).astype(np.float64)
                #b = state of sigma_0 for all the samples
                
                #定数としてbを定義
                inputs=tf.constant(dtype=tf.float64,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
                #Initial input to feed to the rnn


                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                samples=[]

                rnn_state=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for ny in range(self.Ny): #Loop over the lattice in a snake shape
                  for nx in range(self.Nx): 
                    #rnn_output, hidden_layer 
                    rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                    #softmaxをかけてrnn_outputを生成確率っぽくする
                    output=self.dense(rnn_output)
                    #得た生成確率を使って0~nのsampleを生成する(nは各サイトの状態を指定する)
                    sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                    samples.append(sample_temp)
                    #sampleのうち、各サイトの0~nの値をn次元ベクトル(状態ベクトル)に書き換える
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1

        return self.samples

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_x*system_size_y)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space
            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]

            #多分tf.zerosがベクトルしか作れないからこうやってる？
            a=tf.zeros(self.numsamples, dtype=tf.float64)
            b=tf.zeros(self.numsamples, dtype=tf.float64)
            inputs=tf.stack([a,b], axis = 1)

            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                probs=[]

                rnn_state=self.rnn.zero_state(self.numsamples,dtype=tf.float64)

                for ny in range(self.Ny):
                  for nx in range(self.Nx):
                      rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                      output=self.dense(rnn_output)
                      probs.append(output)
                      #tf.slice(samples,begin=[np.int32(0),np.int32(ny*self.Nx+nx)],size=[np.int32(-1),np.int32(1)])
                      #=>sampleの中からあるサイトのインプットだけを抜き出す
                      #それを状態ベクトルに変換する
                      inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(ny*self.Nx+nx)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs