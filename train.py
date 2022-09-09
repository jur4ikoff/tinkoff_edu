import os, sys
import argparse, pymorphy2
import gensim
import numpy as np

inputt = False

# try:
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--inputt', type=str, help='Input path to open', default=False)
#     parser.add_argument('--model', type=str, help='Model', default='\\model')
#     args = parser.parse_args()
# except Exception as e:
#     pass

model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)


def clear_text(text: str):
    symbols_to_delete = ['.', ',', '\'', '!', '(', ')', '?', '"']

    for s in symbols_to_delete:
        text = text.replace(s, '')

    symbols_to_replace = {'ё': 'е'}
    for k, v in symbols_to_replace.items():
        text = text.replace(k, v)

    return text


a = []
tokens = []
try:
    if not inputt:
        for line in sys.stdin:
            line = clear_text(line).rstrip()
            a.append(line.lower())
except KeyboardInterrupt:
    pass


def determine_vector(text: str):
    clean_text = clear_text(text)
    # print(clean_text)
    text_list = clean_text.split(' ')

    text_len = len(text_list)

    morph = pymorphy2.MorphAnalyzer(lang='ru')
    bad = []
    good = []

    good_count = 0
    first_start = True

    for word in text_list:
        word_analyzed = morph.parse(word)[0]
        POS = word_analyzed.tag.POS
        normal_form = word_analyzed.normal_form.replace('ё', 'е')

        pymorph_POS_to_w2v = {'ADVB': 'ADV', 'ADJF': 'ADJ', 'NPRO': 'PROPN',
                              'PRCL': 'NOUN'}
        for k, v in pymorph_POS_to_w2v.items():
            POS = POS.replace(k, v)

        try:
            vec = model.get_vector(normal_form + '_' + str(POS))
            good.append(normal_form + '_' + str(POS))
        except:
            bad.append(normal_form + '_' + str(POS))
            continue

        if first_start:
            text_vec = vec
            first_start = False
            good_count += 1
            continue

        text_vec = np.add(text_vec, vec)
        good_count += 1

    res = text_vec / good_count
    return res


for i in a:
    print(determine_vector(i))

# Сделано - Ввод + аргпарсер. Очистка + токенизация. Опыт в разработке нейронок не имею
