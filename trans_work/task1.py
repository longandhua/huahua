sentences = ["我喜欢你", "我爱他", "我讨厌吃榴莲", "我热爱祖国"]  # 数据集


#  训练
def do_batch(sentences):
    in_batch = []
    out_batch = []
    # 将数据集分词
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    for s in sentences:
        word = s.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        in_batch.append(np.eye(n_class)[input])
        out_batch.append(np.eye(n_class)[target])

    return in_batch, out_batch
