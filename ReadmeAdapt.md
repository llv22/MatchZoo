# Match Zoo Adapation for CIKM AnalytiCup 2018

## 1. Data preparation

Originate from path CIKMAnalytiCup2018/MatchZoo/examples/toy_example/generate_classification_data.sh
Status: https://github.com/NTSC-Community/awaresome-neural-models-for-semantic-match

### Generate matchzoo data for ranking
```bash
cd CIKMAnalytiCup2018/MatchZoo/examples/toy_example/
python test_preparation_for_classify.py
```

### Download embedding
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.50d.txt ../../data/toy_example/classification/
```

### Map word embedding
```bash
python gen_w2v.py ../../data/toy_example/classification/glove.6B.50d.txt ../../data/toy_example/classification/word_dict.txt ../../data/toy_example/classification/embed_glove_d50
python norm_embed.py  ../../data/toy_example/classification/embed_glove_d50 ../../data/toy_example/classification/embed_glove_d50_norm
```

### Run to generate binsum for aNMM
```bash
python test_binsum_generator.py 'classification'
```


## 2. Run train and predict phase
```bash
$ python matchzoo/main.py --phase train --model_file examples/toy_example/config/anmm_classify.config
```