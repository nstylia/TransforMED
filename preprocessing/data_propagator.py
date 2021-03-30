import os

def propagate_data(data_dir,model_dir,data_part_directory,gold_name_relations,gold_name_am):
    'based on the pre'
    gold_am = data_dir+data_part_directory + '/' +gold_name_am
    gold_rc = data_dir + data_part_directory + '/' + gold_name_relations
    pred_am = model_dir+'sequence_tagging_predictions_'+data_part_directory+'.conll'
    with open(pred_am) as f :
        data = f.readlines()
        pred_data = []
        sentence = []
        for line in data:
            line = line.strip('\n').split('\t')
            if len(line)>1:
                word = line[0]
                label = line[2]
                sentence.append([word, label])
            else:
                pred_data.append(sentence)
                sentence = []

    with open(gold_am) as f:
        data = f.readlines()
        gold_data = []
        sentence = []
        for line in data:
            line = line.strip('\n').split('\t')
            if len(line)>1:
                id = line[0]
                word = line[1]
                pico = line[2]
                label = line[3]
                sentence.append([id,word,pico,label])
            else:
                gold_data.append(sentence)
                sentence = []

        #the data splits are very weird and cannot make it to original and comapre at the moment.
        #will come back to it later
    #     will come back later.

    # arguments  = []
    # for sentence in gold_data:
    #     #building sentence
    #     argument_str = ''
    #     argument_pico = []
    #     arguments = []
    #     arg_start = None
    #     arg_end = None
    #     arg_label = None
    #     for i,token in enumerate(sentence):
    #         label = token[-1]
    #         if label != 'O' and arg_label is None: #starting new argument
    #             arg_label = label.split('-')[-1]
    #             arg_start = int(token[0])
    #             argument_str += token[1]+' '
    #             argument_pico.append(token[2])
    #         elif label != 'O' and arg_label is not None: #continue in the same argument
    #             if arg_label in label:
    #                 argument_str += token[1]+' '
    #                 argument_str += token[1]
    #                 argument_pico.append(token[2])
    #         elif label == '0' and arg_label is not None: #end argument
    #             arg_label = None
    #             arg_start = None
    #             arg_end = token[0]-1
    #             arguments.append([argument_str, argument_pico])
    #             argument_str = ''
    #             argument_pico = []
    #         if (len(sentence)-1) == i and arg_label is not None: #whole sentence was argument
    #             arg_label = None
    #             arg_start = None
    #             arg_end = int(token[0]) - 1
    #             arguments.append([argument_str, argument_pico, arg_start, arg_end])
    #             argument_str = ''
    #             argument_pico = []

    gold_sentences = []
    for data in gold_data:
        tokens = []
        for entry in data:
            tokens.append(entry[1])
        gold_sentences.append(' '.join(tokens))

    with open(gold_rc) as f:
        data = f.readlines()
        rc_data = []
        for line in data:
            line = line.strip('\n').split('\t')
            label = line[0]
            sent_a = line[1]
            sent_b = line[2]
            rc_data.append([sent_a,sent_b,label])

    full_data = []
    for rc_entry in rc_data:
        sentence_a = rc_entry[0]
        sentence_a_data = [sentence_a]
        sentence_b = rc_entry[1]
        sentence_b_data = [sentence_b]
        label = rc_entry [2]
        for sentence in gold_sentences:
            if sentence_a in sentence:
                index = gold_sentences.index(sentence)
                pico_a = []
                labels_a = []
                for entry in gold_data[index]:
                    pico_a.append(entry[2])
                    labels_a.append(entry[3])
                sentence_a_data.append(pico_a)
                sentence_a_data.append(labels_a)
            if sentence_b in sentence:
                index = gold_sentences.index(sentence)
                pico_b = []
                labels_b = []
                for entry in gold_data[index]:
                    pico_b.append(entry[2])
                    labels_b.append(entry[3])
                sentence_b_data.append(pico_b)
                sentence_b_data.append(labels_b)
        full_data.append([sentence_a_data,sentence_b_data,label])


    #compare ids based on indexes.
    # percentage_comparator()





    return 0

def main():
    data_dir = '../data/'
    model_dir = '../output/seqtag128_testout/'
    dirs = ['neoplasm', 'glaucoma_test', 'mixed_test']
    # predicted_files = 'sequence_tagging_preddictions_'
    gold_files_relations = 'test_relations.tsv'
    gold_files_am = 'test_agg.conll'
    for dir in dirs:
        propagate_data(data_dir,model_dir,dir,gold_files_relations,gold_files_am)
    # am_files = [
    #     '/home/nikos/projects/ecai2020-transformer_based_am-master/data/glaucoma_test/test_io.',
    #     '/home/nikos/projects/ecai2020-transformer_based_am-master/data/neoplasm/dev_io.',
    #     '/home/nikos/projects/ecai2020-transformer_based_am-master/data/neoplasm/test_io.',
    #     '/home/nikos/projects/ecai2020-transformer_based_am-master/data/neoplasm/train_io.',
    #     '/home/nikos/projects/ecai2020-transformer_based_am-master/data/mixed_test/test_io.']

    return 0

if __name__ == '__main__':
    main()