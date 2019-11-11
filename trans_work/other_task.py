import numpy as np

# 分词和切分词
train_tokens = []
for text in train_texts_orig:
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+--!,.?\~@￥%......&*()]+", "", text)  # 去掉标点
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    train_tokens.append(cut_list)


# 索引长度标准化
def index_standar():
    num_tokens = [len(tokens) for tokens in train_tokens]
    num_tokens = np.array(num_tokens)
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    return max_tokens


# 反向切分词
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ''
    return text


reverse_tokens(train_tokens[0])


# 构建模型
def make_model():
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)
    earlystopping = EerlyStopping(monitor='val_loss', patience=3, verbose=1)
    lr_reduction = ReduceROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
    model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=128, callbacks=callbacks)


# 预测
def predict_sentiment(text):
    print(text)
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+--!,.?\~@￥%......&*()]+", "", text)
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价', 'output = %2.f' % coef)
    else:
        print('是一例负面评价', 'output = %2.f' % coef)


# 测试
idx = 101
print(reverse_tokens(X_test[idx]))
print('预测的分类', y_pred[idx])
print('实际的分类', y_actual[idx])
