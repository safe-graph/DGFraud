import numpy as np
import tensorflow as tf 
SIZE=6
CLASS=8
label1=tf.constant([0,1,2,3,4,5,6,7])
sess1=tf.Session()
print('label1:',sess1.run(label1))
b = tf.one_hot(label1,depth=CLASS,axis=0)
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())    
    sess.run(b)    
    print('after one_hot',sess.run(b))
