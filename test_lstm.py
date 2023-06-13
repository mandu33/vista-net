import os
import gpu_utils

gpu_utils.setup_no_gpu()

import tensorflow as tf
from datetime import datetime

from data_reader import DataReader
from data_preprocess import cities

from model import VistaNet
from model_utils import count_parameters
#调用sklearn计算指标
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

#参数
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", 'checkpoints',
                       """Path to checkpoint folder""")
tf.flags.DEFINE_string("log_dir", 'log',
                       """Path to log folder""")

tf.flags.DEFINE_integer("num_checkpoints", 1,
                        """Number of checkpoints to store (default: 1)""")
tf.flags.DEFINE_integer("num_epochs", 20,
                        """Number of training epochs (default: 10)""")
tf.flags.DEFINE_integer("batch_size", 32,
                        """Batch Size (default: 32)""")
tf.flags.DEFINE_integer("display_step", 20,
                        """Display after number of steps (default: 20)""")

tf.flags.DEFINE_float("learning_rate", 0.001,
                      """Learning rate (default: 0.001)""")
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      """Maximum value for gradient clipping (default: 5.0)""")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      """Probability of keeping neurons (default: 0.5)""")

tf.flags.DEFINE_integer("hidden_dim", 50,
                        """Hidden dimensions of GRU cell (default: 50)""")
tf.flags.DEFINE_integer("att_dim", 100,
                        """Attention dimensions (default: 100)""")
tf.flags.DEFINE_integer("emb_size", 200,
                        """Word embedding size (default: 200)""")
tf.flags.DEFINE_integer("num_images", 3,
                        """Number of images per review (default: 3)""")
tf.flags.DEFINE_integer("num_classes", 5,
                        """Number of classes of prediction (default: 5)""")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

tf.flags.DEFINE_boolean("use_lstm", True,
                        """Use LSTM instead of RNN""")
train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
valid_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')


valid_step = 0

def evaluate(session, dataset, model, loss, accuracy, error, precision, recall, summary_op=None):
  sum_loss = 0.0
  sum_acc = 0.0
  sum_err = 0.0
  sum_pre = 0.0
  sum_rec = 0.0
  example_count = 0
  global valid_step
  
  for reviews, images, labels in dataset:
    feed_dict = model.get_feed_dict(reviews, images, labels)
    _loss, _acc, _err, _pre, _rec = session.run([loss, accuracy, error, precision, recall], feed_dict=feed_dict)
    if not summary_op is None:
      _summary = session.run(summary_op, feed_dict=feed_dict)
      valid_summary_writer.add_summary(_summary, global_step=valid_step)
      valid_step += len(labels)

    sum_loss += _loss * len(labels)
    sum_acc += _acc * len(labels)
    sum_err += _err * len(labels)
    sum_pre += _pre * len(labels)
    sum_rec += _rec * len(labels)
    example_count += len(labels)

  avg_loss = sum_loss / example_count
  avg_acc = sum_acc / example_count
  avg_err = sum_err / example_count
  avg_pre = sum_pre / example_count
  avg_rec = sum_rec / example_count
  avg_f1 = 2*avg_pre*avg_rec/(avg_pre+avg_rec)
  return avg_loss, avg_acc, avg_err, avg_pre, avg_rec, avg_f1


def test(session, data_reader, model, loss, accuracy, error, precision, recall, epoch, result_file):
  for city in cities:
    #accuracy,error
    test_loss, test_acc, test_error, test_pre, test_rec, test_f1 = evaluate(session, data_reader.read_test_set(city), model, loss, accuracy, error, precision, recall)
    #test_f1 = 2*test_pre*test_rec/(test_pre+test_rec)
    result_file.write('city={},epoch={},loss={:.4f},acc={:.4f},error={:.4f},pre={:.4f},rec={:.4f},f1={:.4f}\n'.format(city, epoch, test_loss, test_acc, test_error, test_pre, test_rec, test_f1))
  result_file.flush()
 

def train(session, data_reader, model, train_op, loss, accuracy, error, precision, recall, summary_op):
  for reviews, images, labels in data_reader.read_train_set(batch_size=FLAGS.batch_size):
    step, _, _loss, _acc, _err, _pre, _rec = session.run([model.global_step, train_op, loss, accuracy, error, precision, recall],
                                       feed_dict=model.get_feed_dict(reviews, images, labels,
                                                                     FLAGS.dropout_keep_prob))
    if step % FLAGS.display_step == 0:
      _summary = session.run(summary_op, feed_dict=model.get_feed_dict(reviews, images, labels,
                                                                       dropout_keep_prob=1.0))
      

def loss_fn(labels, logits):
  onehot_labels = tf.one_hot(labels, depth=FLAGS.num_classes)
  cross_entropy_loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels,
    logits=logits
  )
  tf.summary.scalar('loss', cross_entropy_loss)
  return cross_entropy_loss


def train_fn(loss, global_step):
  trained_vars = tf.trainable_variables()
  count_parameters(trained_vars)

  # Gradient clipping
  gradients = tf.gradients(loss, trained_vars)
  clipped_grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
  tf.summary.scalar('global_grad_norm', global_norm)

  optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
  train_op = optimizer.apply_gradients(zip(clipped_grads, trained_vars),
                                       name='train_op',
                                       global_step=global_step)
  return train_op

#accuracy=(TP+TN)/(TP+TN+FP+FN)
def eval_fn(labels, logits):
  prediction = tf.argmax(logits, axis=-1)#预测值
  corrected_pred = tf.equal(prediction, tf.cast(labels, tf.int64))
  accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  return accuracy
#error=1-accuracy
def error_fn(labels, logits):
    prediction = tf.argmax(logits, axis=-1)
    incorrect_pred = tf.not_equal(prediction, tf.cast(labels, tf.int64))
    error_rate = tf.reduce_mean(tf.cast(incorrect_pred, tf.float32))
    tf.summary.scalar('error_rate', error_rate) # define summary
    return error_rate
#recall--problem
#recall=TP/(TP+FN)
def recall_fn(labels, logits):
  prediction = tf.argmax(logits, axis=-1)
  #TP
  true_positives = tf.math.count_nonzero(prediction * tf.cast(labels, tf.int64))
  #标签为正类的数量
  labeled_positives = tf.math.count_nonzero(labels)
  recall = tf.math.divide(true_positives, labeled_positives)
  tf.summary.scalar('recall', recall) # define summary
  return recall
#precision=TP/(TP+FP)
def precision_fn(labels, logits):
    prediction = tf.argmax(logits, axis=-1)
    #TP
    true_positives = tf.math.count_nonzero(prediction * tf.cast(labels, tf.int64))
    #预测为正类的数量
    predicted_positives = tf.math.count_nonzero(prediction)
    precision = tf.math.divide(true_positives, predicted_positives)
    tf.summary.scalar('precision', precision) # define summary
    return precision

#指标计算
#error=1-accuracy
# def error_fn(labels, logits):
#   prediction = tf.argmax(logits, axis=-1)#最大值
#   start = tf.equal(prediction, tf.cast(labels, tf.int64))
#   between = tf.cast(tf.zeros_like(start),tf.bool)
#   after = tf.equal(between, start)
#   with tf.Session() as sess:
#     after = sess.run(after)
#   error = tf.reduce_mean(tf.cast(after, tf.float32))
#   tf.summary.scalar('error', error_fn)
#   return error
def report(labels,logits):
  prediction = tf.argmax(logits, axis=-1)#最大值
  measure_result = classification_report(labels, prediction)#输出模型评估报告
  print("------repost------")
  print(measure_result)
  #包含了五种数据集的各个指标报告，需要进行进一步的平均处理
  #precison
  precision_score_average_None = precision_score(lables, prediction, average=None)
  with open("./loss/precision_test.txt", 'w') as file1:
          file1.write(str(precision_score_average_None))
  #recall
  recall_score_average_None = recall_score(lables, prediction, average=None)
  with open("./loss/recall_test.txt", 'w') as file2:
          file2.write(str(recall_score_average_None))
  #F1 score
  f1_score_average_None = f1_score(lables, prediction, average=None)
  with open("./loss/F1_test.txt", 'w') as file3:
          file3.write(str(f1_score_average_None))




def main(_):
  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=config) as sess:
    print('\n{} Model initializing'.format(datetime.now()))

    model = VistaNet(FLAGS.hidden_dim, FLAGS.att_dim, FLAGS.emb_size, FLAGS.num_images, FLAGS.num_classes, FLAGS.use_lstm)
    loss = loss_fn(model.labels, model.logits)
    train_op = train_fn(loss, model.global_step)
    #五个指标
    accuracy = eval_fn(model.labels, model.logits)
    error = error_fn(model.labels, model.logits)
    precision = precision_fn(model.labels, model.logits)
    recall = recall_fn(model.labels, model.logits)
    #f1在train()内部计算
    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    train_summary_writer.add_graph(sess.graph)
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
    data_reader = DataReader(num_images=FLAGS.num_images, train_shuffle=True)

    print('\n{} Start training'.format(datetime.now()))

    epoch = 0
    best_loss = float('inf')
    #数组保存loss进行初始化
    train_loss_result = []
    valid_loss_result = []
    while epoch < FLAGS.num_epochs:
      epoch += 1
      print('\n=> Epoch: {}'.format(epoch))
      
      train(sess, data_reader, model, train_op, loss, accuracy, error, precision, recall, summary_op)
      print('=> Evaluation')
      print('best_loss={:.4f}'.format(best_loss))
      

      train_loss_result.append(best_loss)
      valid_loss, valid_acc, valid_err, valid_pre, valid_rec, valid_f1 = evaluate(sess, data_reader.read_valid_set(batch_size=FLAGS.batch_size), model, loss, accuracy, error, precision, recall, summary_op)
      valid_f1 = 2*valid_pre*valid_rec/(valid_pre+valid_rec)
      print('valid_loss={:.4f}, valid_acc={:.4f}, valid_err={:.4f},valid_pre={:.4f}, valid_rec={:.4f}, valid_f1={:.4f}'.format(valid_loss, valid_acc, valid_err, valid_pre, valid_rec, valid_f1))
      valid_loss_result.append(valid_loss)

      if valid_loss < best_loss:
        best_loss = valid_loss
        save_path = os.path.join(FLAGS.checkpoint_dir,
                                 'epoch={}-loss={:.4f}-acc={:.4f}-err={:.4f}-pre={:.4f}-rec={:.4f}-f1={:.4f}'.format(epoch, valid_loss, valid_acc, valid_err, valid_pre, valid_rec, valid_f1))
        saver.save(sess, save_path)
        print('Best model saved @ {}'.format(save_path))

    # print('=> Testing')
    # result_file = open(os.path.join(FLAGS.log_dir, 'loss={:.4f}, acc={:.4f}, err={:.4f}, pre={:.4f}, rec={:.4f}, f1={:.4f}, epoch={}'.format(valid_loss, valid_acc, valid_err, valid_pre, valid_rec, valid_f1, epoch)), 'w')
    # test(sess, data_reader, model, loss, accuracy, error, precision, recall, epoch, result_file)
    print('=> Testing')
    result_file = open(os.path.join(FLAGS.log_dir, 'loss={:.4f}, acc={:.4f}, err={:.4f}, pre={:.4f}, rec={:.4f}, f1={:.4f}, epoch={}'.format(valid_loss, valid_acc, valid_err, valid_pre, valid_rec, valid_f1, epoch)), 'w')
    cities=["Boston","Chicago","Los Angeles","New York","San Francisco"]
    for city in cities:
      test_loss, test_acc, test_err, test_pre, test_rec, test_f1 = evaluate(sess, data_reader.read_test_set(city,batch_size=FLAGS.batch_size), model, loss, accuracy, error, precision, recall, summary_op)
      test_f1 = 2*test_pre*test_rec/(test_pre+test_rec)
      print('city={},epoch={},loss={:.4f},acc={:.4f},error={:.4f},pre={:.4f},rec={:.4f},f1={:.4f}\n'.format(city, epoch, test_loss, test_acc, test_err, test_pre, test_rec, test_f1))
      result_file.write('city={},epoch={},loss={:.4f},acc={:.4f},error={:.4f},pre={:.4f},rec={:.4f},f1={:.4f}\n'.format(city, epoch, test_loss, test_acc, test_err, test_pre, test_rec, test_f1))
    result_file.flush()
        
  
  with open("./loss/lstm/train_lstm_loss.txt", 'w') as train_loss_file:
          train_loss_file.write(str(train_loss_result))
  with open("./loss/lstm/valid_lstm_loss.txt", 'w') as valid_loss_file:
          valid_loss_file.write(str(valid_loss_result))
  print("{} Optimization Finished!".format(datetime.now()))

if __name__ == '__main__':
  tf.app.run()
