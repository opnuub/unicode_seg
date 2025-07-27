
def get_embedding():
    from transformers import TFBertModel
    from cihai.core import Cihai
    import tensorflow as tf
    import numpy as np
    import lists

    c = Cihai()
    if not c.unihan.is_bootstrapped:
        c.unihan.bootstrap()

    sums = {}
    counts = {}
    tfm  = TFBertModel.from_pretrained("uer/chinese_roberta_L-2_H-128")
    emb  = tfm.bert.embeddings.weight           # [21128,128] tensor
    
    processed = []
    with open ('vocab.txt', 'r') as f:
        lst = f.readlines()
        processed = [i.strip('\n').strip('##') for i in lst]

    for j in range(len(processed)):
        i = processed[j]
        char = c.unihan.lookup_char(i).first()
        if not char:
            continue
        radical = char.kRSUnicode.split(" ")
        if len(radical) == 1:
            z = radical[0]
            if '\'' in z:
                num = z.split('\'')[0]
            else:
                num = z.split('.')[0]
            if num not in sums:
                sums[num] = emb[j]
            else:
                sums[num] += emb[j]
            counts[num] = counts.get(num, 0) + 1

    for key, value in sums.items():
        sums[key] = value / counts[key]

    for i in lists.map:
        try:
            index = processed.index(i)
            sums[index] = emb[index]
        except:
            pass

    sigma = tf.math.reduce_std(emb, axis=0)
    mu    = tf.reduce_mean(emb, axis=0)

    for i in range(len(lists.cantonese_only)):
        j = lists.cantonese_only[i]
        index = str(-1-i)
        z = tf.random.normal([128]) * sigma * 1.0 + mu
        sums[index] = z.numpy()

    # radicals 1-214
    lst = [tfm.bert.embeddings.weight[0]]
    for i in range(1, 215):
        radical_index = str(i)
        vector = sums.get(radical_index, mu)
        lst.append(vector)
    # constants 215 - 249
    for i in range(1, 36):
        constant_index = str(0-i)
        lst.append(sums[constant_index])
    # single-form 250 - 544
    single = {}
    j = 0
    for i in lists.map:
        try:
            index = processed.index(i)
            single[i] = j+249
            j += 1
            lst.append(emb[index])
        except:
            pass

    #emb_init = tf.keras.initializers.Constant(lst)
    mat = np.squeeze(np.array([lst]), axis=0)
    np.save("cantonese_emb.npy", mat)

get_embedding()