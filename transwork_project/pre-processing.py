# 预处理数据
def split_to_token(sent, is_source):
    global src_unk_count, tgt_unk_count
    sent = sent.replace(',', ' ,')
    sent = sent.replace('.', ' .')
    sent = sent.replace('\n', ' ')
    sent_toks = sent.split(' ')
    for t_i, tok in enumerate(sent_toks):
        if is_source:
            if tok not in src_dictionary.keys():
                sent_toks[t_i] = '<unk>'
                src_unk_count += 1
        else:
            if tok not in tgt_dictionary.keys():
                if not len(tok.strip()) == 0:
                    sent_toks[t_i] = '<unk>'
                    tgt_unk_count += 1
        return sent_toks
