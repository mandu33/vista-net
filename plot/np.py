import numpy as np
#数组处理
# test = []
# for i in range(1,10):
#     test.append(float(i))
# print(test)
#tf bool取反
import tensorflow as tf
a = [[1,2,3],[4,5,6]]
b = [[1,0,3],[1,5,1]]
with tf.Session() as sess:
    start = sess.run(tf.equal(a,b))
print("start:")
print(start)
print("change......")
# between = tf.cast(tf.zeros_like(start),tf.bool)
# after = tf.equal(between, start)
after = tf.logical_not(start)
print("after:")
with tf.Session() as sess:
    after = sess.run(after)
print(after)