import os


'''
Simple formatter to remove '-DOCSTART- -X- O O\n' from the data created from EBM-NLP dataset.
With this it creates uniform data with the neoplasm/glaucoma/mixed splits and can be used with
the same functions. Run this once, if the data are created using the scripts provided with the
EBM-NLP dataset, after creating the all annotations non-hierarcical train/dev/gold splits.
'''

def main():
    pico_files = [
        '/data/pico_ner/p1_all_dev.txt',
        '/data/pico_ner/p1_all_gold.txt',
        '/data/pico_ner/p1_all_train.txt',
    ]
    root = os.getcwd()

    match_string = '-DOCSTART- -X- O O\n'

    for pico_file in pico_files:
        file = os.path.join(root, pico_file)
        with open(file, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line == match_string:
                lines.pop(i)
                if lines[i] == '\n':
                    lines.pop(i)
        with open(pico_file, 'w') as f:
            for line in lines:
                f.write(line)

if __name__ == '__main__':
    main()