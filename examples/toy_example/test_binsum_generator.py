# /bin/python2.7
import os
import sys
import numpy as np
# Preprocessing library import
if os.path.exists('../../matchzoo/inputs') and os.path.exists('../../matchzoo/utils'):
    sys.path.append('../../matchzoo/inputs')
    sys.path.append('../../matchzoo/utils')
else:
    assert os.path.exists('matchzoo/inputs') and os.path.exists('matchzoo/utils')
    sys.path.append('matchzoo/inputs')
    sys.path.append('matchzoo/utils')
from preprocess import *
from rank_io import *

if __name__ == '__main__':
    run_mode = 'ranking'
    if len(sys.argv) > 1 and sys.argv[1] == 'classification':
        run_mode = 'classification'
    bin_num = 30
    # Fixed relative path issue
    if os.path.exists('../../data/toy_example/%s/'%(run_mode)):
        path = '../../data/toy_example/%s/'%(run_mode)
    else:
        assert os.path.exists('data/toy_example/%s/'%(run_mode))
        path = 'data/toy_example/%s/'%(run_mode)
    embed_size = 50
    embedfile = path + 'embed_glove_d50_norm'
    corpfile = path + 'corpus_preprocessed.txt'
    relfiles = [path + 'relation_train.txt',path + 'relation_valid.txt',path + 'relation_test.txt']
    # embedding layer is decoupled with bin_num variable
    histfiles = [path + 'relation.train.binsum-%d.txt'%(bin_num),path + 'relation.valid.binsum-%d.txt'%(bin_num), path + 'relation.test.binsum-%d.txt'%(bin_num)]

    # note here word embeddings have been normalized to speed up calculation
    embed_dict = read_embedding(filename = embedfile)
    print('after read embedding ...')
    # for word without wordembeeding, assign an random embedding
    _PAD_ = len(embed_dict)
    embed_dict[_PAD_] = np.zeros((embed_size, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [_PAD_+1, embed_size]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    corp, _ = read_data(corpfile)
    print('after read corpus ....')

    # Generate binsum-30 for train, valid and test dataset
    for i in range(len(relfiles)):
        rel = read_relation(relfiles[i])
        fout = open(histfiles[i], 'w')
        inum = 0
        for label, d1, d2 in rel:
            inum += 1
            assert d1 in corp
            assert d2 in corp
            qnum = len(corp[d1])
            d1_embed = embed[corp[d1]]
            d2_embed = embed[corp[d2]]
            curr_binsum = cal_binsum(d1_embed, d2_embed, qnum, bin_num)
            curr_binsum = curr_binsum.tolist()
            fout.write(' '.join(map(str, curr_binsum)))
            fout.write('\n')
            if inum % 1000 == 0:
                print('inum: %d ....\r'%inum,)
                sys.stdout.flush()
            #print(curr_hist)
        fout.close()
        print('file: %s processed... '%(relfiles[i]))
    print('\nfinished ...')
