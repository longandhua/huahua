# -------------------------细胞计算------------------------

# 细胞计算（编码器）的定义
def enc_lstm_cell(i, o, state):
    # 创建一个LSTM细胞
    input_gate = tf.sigmoid(tf.matmul(i, enc_ix) + tf.matmul(o, enc_im) + enc_ib)
    forget_gate = tf.sigmoid(tf.matmul(i, enc_fx) + tf.matmul(o, enc_fm) + enc_fb)
    update = tf.matmul(i, enc_cx) + tf.matmul(o, enc_cm) + enc_cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, enc_ox) + tf.matmul(o, enc_om) + enc_ob)
    return output_gate * tf.tanh(state), state


# 细胞计算（解码器）的定义
def dec_lstm_cell(i, o, state):
    # 创建一个LSTM细胞
    input_gate = tf.sigmoid(tf.matmul(i, dec_ix) + tf.matmul(o, dec_im) + dec_ib)
    forget_gate = tf.sigmoid(tf.matmul(i, dec_fx) + tf.matmul(o, dec_fm) + dec_fb)
    update = tf.matmul(i, dec_cx) + tf.matmul(o, dec_cm) + dec_cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, dec_ox) + tf.matmul(o, dec_om) + dec_ob)
    return output_gate * tf.tanh(state), state

# -------------------------训练------------------------------
# 训练逻辑
outputs = list()
output = saved_output
state = saved_state
print('Calculating Encoder Output')
for i in enc_train_inputs:
 output, state = enc_lstm_cell(i, output,state)
print('Calculating Decoder Output')
with tf.control_dependencies([saved_output.assign(output),saved_state.assign(state)]):
    for i in dec_train_inputs:
      output, state = dec_lstm_cell(i, output, state)
      outputs.append(output)
logits = tf.matmul(tf.concat(axis=0, values=outputs), w) + b
train_prediction = tf.nn.softmax(logits)

# ---------------------------------测试--------------------------
test_output  = saved_test_output
test_state = saved_test_state
test_predictions = []
for i in enc_test_input:
  test_output, test_state = enc_lstm_cell(i, test_output,test_state)
with tf.control_dependencies([saved_test_output.assign(test_output),saved_test_state.assign(test_state)]):
    for i in range(dec_num_unrollings):
       test_output, test_state = dec_lstm_cell(dec_test_input, test_output, test_state)
       test_prediction = tf.nn.softmax(tf.nn.xw_plus_b(test_output, w, b))
       dec_test_input = tf.nn.embedding_lookup(tgt_word_embeddings,tf.argmax(test_prediction,axis=1))
       test_predictions.append(tf.argmax(test_prediction,axis=1))

# ----------------------损失计算----------------------
loss_batch = tf.concat(axis=0, values=dec_train_masks) * tf.nn.softmax_cross_entropy_with_logits_v2( logits = logits, labels = tf.concat(axis=0, values=dec_train_labels))

loss = tf.reduce_mean(loss_batch)

# ------------------------------Adam更换为SGD--------------------------------
with tf.variable_scope('Adam'):
   optimizer = tf.train.AdamOptimizer(learning_rate)
with tf.variable_scope('SGD'):
   sgd_optimizer = tf.train.GradientDescentOptimizer(sgd_learning_rate)
# 利用裁剪方法计算Adam的梯度  ,
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimize = optimizer.apply_gradients(zip(gradients, v))
# 利用裁剪方法计算SGD的梯度  ,
sgd_gradients, v = zip(*sgd_optimizer.compute_gradients(loss))
sgd_gradients, _ = tf.clip_by_global_norm(sgd_gradients, 5.0)
sgd_optimize = optimizer.apply_gradients(zip(sgd_gradients, v))
# 确保从解码器到编码器的梯度存在  ,
for (g_i,v_i) in zip(gradients,v):
   assert g_i is not None, ' Gradient none for %s '%(v_i.name)

# ------------------------------运行神经网络机器翻译系统---------------------------------
def print_and_save_train_predictions(du_labels, tr_pred, rand_idx, train_prediction_text_fname):
     print_str = 'Actual: '
     for w in np.argmax(np.concatenate(du_labels,axis=0)[rand_idx::batch_size],axis=1).tolist():
         print_str += tgt_reverse_dictionary[w] + ' '
         if tgt_reverse_dictionary[w] == '</s>':
             break
     print(print_str)
     with open(os.path.join(log_dir, train_prediction_text_fname),'a',encoding='utf-8') as fa:
         fa.write(print_str+'\ ')
     print()
     print_str = 'Predicted: '
     for w in np.argmax(tr_pred[rand_idx::batch_size],axis=1).tolist():
         print_str += tgt_reverse_dictionary[w] + ' '

         if tgt_reverse_dictionary[w] == '</s>':
             break
     print(print_str)
     with open(os.path.join(log_dir, train_prediction_text_fname),'a',encoding='utf-8') as fa:
          fa.write(print_str+'\ ')
          ve_test_predictions(test_du_labels, test_pred_unrolled, batch_id, test_rand_idx, test_prediction_text_fname):
     print('DE: ',test_source_sent[(batch_id*batch_size)+test_rand_idx])
     print_str = 'EN (TRUE):' + test_target_sent[(batch_id*batch_size)+test_rand_idx]

     print_str = 'EN (Predicted): '
     for test_pred in test_pred_unrolled:
         print_str += tgt_reverse_dictionary[test_pred[test_rand_idx]] + ' '
         if tgt_reverse_dictionary[test_pred[test_rand_idx]] == '</s>':
             break
     print(print_str + '\ ')
     with open(os.path.join(log_dir, test_prediction_text_fname),'a',encoding='utf-8') as fa:
        fa.write(print_str+'\ ')

def create_bleu_ref_candidate_lists(all_preds, all_labels):
     bleu_labels, bleu_preds = [],[]
     ref_list, cand_list = [],[]
     for b_i in range(batch_size):
         tmp_lbl = all_labels[b_i::batch_size]
         tmp_lbl = tmp_lbl[np.where(tmp_lbl != tgt_dictionary['</s>'])]
         ref_str = ' '.join([tgt_reverse_dictionary[lbl] for lbl in tmp_lbl])
         ref_list.append([ref_str])
         tmp_pred = all_preds[b_i::batch_size]
         tmp_pred = tmp_pred[np.where(tmp_pred != tgt_dictionary['</s>'])]
         cand_str = ' '.join([tgt_reverse_dictionary[pre] for pre in tmp_pred])
         cand_list.append(cand_str)
     return cand_list, ref_list
