import os

'''
Changes annotation scheme from IOB to IO.
Adds the PICO annotations created from the EBM model to the Argument Mining annotations, 
so that you don't need to use the EBM model to get annotations live during training and inference. 
Finally, shows some statistics about the PICO annotations in the aggregated files with respect to 
the Argument Relation classes. 
'''

def make_io(file):
    print('Converting IOB to IO for file: ' + file )
    with open(file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n').split('\t')

    for line in lines:
        if 'B-' in line[-1]:
            tag = line[-1].split('-')[-1]
            line[-1] = 'I-' + tag

    newfile = file.split(".conll")
    newfile = '_io.conll'.join(newfile)
    print('Saving to: ' + newfile)
    with open(newfile, "w+") as f:
        for line in lines:
            f.write('\t'.join(line)+'\n')

    return True

def aggregate(io_file, pico_file):
    newfile = io_file.split('_io.conll')[0]+'_agg.conll'
    with open(io_file) as f:
        io_lines = f.readlines()
        for i in range(len(io_lines)):
            io_lines[i] = io_lines[i].strip('\n').split('\t')
        j=0
        while len(io_lines)!= j:
            if io_lines[j] == ['']:
                if io_lines[j-1] == ['']:
                    io_lines.pop(j)
            j+=1

    with open(pico_file) as f:
        pico_lines = f.readlines()
        for i in range(len(pico_lines)):
            pico_lines[i] = pico_lines[i].strip('\n').split(' ')

    contents = []
    for i in range(0,len(pico_lines)):
        if len(io_lines[i])>1:
            counter = io_lines[i][0]
            word = io_lines[i][1]
            pico_tag = pico_lines[i][2]
            am_tag = io_lines[i][-1]
            line = '\t'.join([counter,word,pico_tag,am_tag])
            contents.append(line+'\n')
        else:
            contents.append('\n')
    if len(pico_lines) < len(io_lines):
        for i in range(len(pico_lines),len(io_lines)):
            counter = io_lines[i][0]
            word = io_lines[i][1]
            pico_tag = 'N'
            am_tag = io_lines[i][-1]
            line = '\t'.join([counter, word, pico_tag, am_tag])
            contents.append(line + '\n')

    with open(newfile, 'w') as f:
        for line in contents:
            f.write(line)
    return newfile

def statistics(agg_file):
    print('Drawing statistics for file: {}'.format(agg_file.split('/')[-1]))
    with open(agg_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip('\n').split('\t')

    sentences = []
    sentence = []
    for line in lines:
        if len(line)>1:
            sentence.append(line)
        else:
            sentences.append(sentence)
            sentence = []

    all_stats = []
    for sentence in sentences:
        am_sentence = set()
        pico_sentence = set()
        for entry in sentence:
            am_sentence.update([entry[-1]])
            pico_sentence.update([entry[-2]])
        all_stats.append([am_sentence,pico_sentence])

    stats = {}
    for sentence in all_stats:
        for am_type in sentence[0]:
            for pico_ent in sentence[1]:
                try:
                    stats[am_type]['count'] = stats[am_type]['count']+1
                except:
                    stats[am_type] = {'count': 1}
                try:
                    stats[am_type][pico_ent] = stats[am_type][pico_ent] +1
                except:
                    stats[am_type][pico_ent] = 1

    pico_arg_occs = 0
    not_pico_arg_occs = 0
    not_pico_sents = []
    for i, sentence in enumerate(all_stats):
        am_type = sentence[0]
        pico = sentence[1]
        if 'O' not in am_type: #am_type != set('O') or
            if pico != set('N'):
                pico_arg_occs+=1
            else:
                not_pico_arg_occs+=1
                not_pico_sents.append(sentences[i])

    print('Arguments with PICO annotations: {}'.format(pico_arg_occs))
    print('Arguments with no PICO annotations: {}'.format(not_pico_arg_occs))

    for am_label in stats:
        print('Sentences with class: {} '.format(am_label))
        print('\t Total occs: {}'.format(stats[am_label]['count']))
        for pico_ents in stats[am_label]:
            print('\t \t {} : {} \t {}%'.format(pico_ents,stats[am_label][pico_ents],(stats[am_label][pico_ents]/stats[am_label]['count'])))

    return 0



def main():
    for dirpath, dirnames, dirfiles in os.walk('../data/'):
        for dirfile in dirfiles:
            if dirfile in ['train.conll','test.conll','dev.conll']:
                file = dirfile.split('.conll')[0]
                io = os.path.join(dirpath,(file+'_io.conll'))
                pico = os.path.join(dirpath,(file+'_io.pico.conll'))
                agg_file = aggregate(io,pico)
                #some stats per file, regarding the labels annotated. 
                statistics(agg_file)
    return 0

if __name__ == "__main__":
    main()