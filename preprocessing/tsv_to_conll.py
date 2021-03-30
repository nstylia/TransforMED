import os
from nltk.tokenize import wordpunct_tokenize
from transformers import BertTokenizer

def bert_transfomer(sentence,pico,tokenizer):
    bert_split = tokenizer.tokenize(sentence)
    def_split = sentence.lower().split(' ')
    for i, token in enumerate(def_split):
        if not token.isalpha() and not token.isnumeric():
            pico.insert(i, pico[i-1])
            if '-' in token:
                pico.insert(i, pico[i - 1])
    for i, token in enumerate(bert_split):
        if '##' in token:
            pico.insert(i, pico[i-1])
    assert(len(pico) == len(bert_split))
    return sentence, ' '.join(pico)

def transform_file(filename):
    print('Transforming: ' + filename)
    output_dir = filename.split('.')[0]+'_for_ner.io'
    with open(filename , 'r') as f:
        data = f.readlines()
        rc_data = []
        for line in data:
            line = line.strip('\n').split('\t')
            label = line[0]
            sent_a = line[1]
            sent_b = line[2]
            rc_data.append([sent_a,sent_b,label])

    new_data = []
    for data in rc_data:
        sentence_a = data[0].split(' ')
        sentence_b = data[1].split(' ')
        new_data.append(sentence_a)
        new_data.append(sentence_b)

    print("Saving at: "+ output_dir)
    with open(output_dir, 'w') as f:
        for data in new_data:
            for entry in data:
                f.write(' '.join([entry, 'POS', 'N\n']))
            f.write('\n')

def convert_to_tsv(filename):
    root = filename.split('_for_ner.iopico.conll')[0]
    original = root+'_for_ner.io'
    tsv = root+'.tsv'
    output = root+'_pico.tsv'

    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


    with open(filename) as f:
        lines = f.readlines()
        sentences = []
        words = []
        tags = []
        for line in lines:
            line = line.strip('\n').split(' ')
            if len(line)>1:
                word = line[0]
                tag = line[2]
                words.append(word)
                tags.append(tag)
            else:
                sentence = ' '.join(words)
                sentences.append([sentence, tags])
                words = []
                tags = []

    with open(tsv) as f:
        data = f.readlines()
        rc_data = []
        sent_data = []
        for line in data:
            line = line.strip('\n').split('\t')
            label = line[0]
            sent_a = line[1]
            sent_b = line[2]
            rc_data.append([sent_a, sent_b, label])
            sent_data.append(sent_a)
            sent_data.append(sent_b)


    try:
        print('Sanity check:')
        assert(len(sent_data) == len(sentences))
        print('Passed')
    except AssertionError:
        print('Assertion failed. File sizes do not match.')
        return 0

    print('Constructing file: ' + output)
    with open(output, 'w') as f:
        for j, i in enumerate(range(0,len(sentences), 2)):
            sentence_a = sent_data[i]
            sentence_b = sent_data[i+1]

            pico_a = ' '.join(sentences[i][-1])
            pico_b = ' '.join(sentences[i+1][-1])


            label = rc_data[j][2]
            f.write('\t'.join([label, sentence_a, sentence_b, pico_a, pico_b])+'\n')

    return 0

def main():
    files = [
        '/data/neoplasm/train_relations.tsv',
        '/data/neoplasm/dev_relations.tsv',
        '/data/neoplasm/test_relations.tsv',
        '/data/glaucoma_test/test_relations.tsv',
        '/data/mixed_test/test_relations.tsv'
    ]

    end_files = [
        '/data/neoplasm/train_relations_for_ner.iopico.conll',
        '/data/neoplasm/dev_relations_for_ner.iopico.conll',
        '/data/neoplasm/test_relations_for_ner.iopico.conll',
        '/data/glaucoma_test/test_relations_for_ner.iopico.conll',
        '/data/mixed_test/test_relations_for_ner.iopico.conll'
    ]

    root = os.getcwd()

    for filename in files:
        file = os.path.join(root,filename) 
        transform_file(file)

    for filename in end_files:
        file = os.path.join(root,filename)
        convert_to_tsv(file)
    return 0

if __name__=='__main__':
    main()