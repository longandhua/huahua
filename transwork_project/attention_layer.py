# 变量的定义
W_a = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.05), name='W_a')
U_a = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.05), name='U_a')
v_a = tf.Variable(tf.truncated_normal([num_nodes, 1], stddev=0.05), name='v_a')


def attn_layer(h_j_unrolled, s_i_minus_1):
    enc_logits = tf.concat(axis=0, values=h_j_unrolled)
    # W_a . encoder_output
    w_a_mul_s_i_minus_1 = tf.matmul(enc_logits, W_a)
    u_a_mul_h_j = tf.matmul(tf.tile(s_i_minus_1, [enc_num_unrollings, 1]), U_a)
    e_j = tf.matmul(tf.nn.tanh(w_a_mul_s_i_minus_1 + u_a_mul_h_j), v_a)
    batched_e_j = tf.split(axis=0, num_or_size_splits=enc_num_unrollings, value=e_j)
    reshaped_e_j = tf.concat(axis=1, values=batched_e_j)
    alpha_i = tf.nn.softmax(reshaped_e_j)
    alpha_i_list = tf.unstack(alpha_i, axis=1)
    c_i_list = [tf.reshape(alpha_i_list[e_i], [-1, 1]) * h_j_unrolled[e_i] for e_i in range(enc_num_unrollings)],
    c_i = tf.add_n(c_i_list)  # of size [batch_size, num_nodes]   ,
    return c_i, alpha_i
