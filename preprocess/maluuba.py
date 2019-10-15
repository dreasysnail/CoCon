# maluuba
import re
import numpy as np

def preprocess():
	correct = dict()
	for line in open('map.tsv', encoding='utf-8'):
		a, b = line.strip('\n').split('\t')
		correct[a.lower().strip()] = b.lower().strip()

	lines = []
	n = 0
	for line in open(DIR+'/raw.txt', encoding='utf-8'):
		line = line.strip('\n').replace("' "," '")
		turns = []
		for turn in line.split('\t'):
			turn = re.sub(r"[^A-Za-z0-9():,\.!?' ]", "", turn)
			aa = turn.split()
			bb = []
			for a in aa:
				bb.append(correct.get(a, a))
			turns.append(' '.join(bb))
		lines.append('\t'.join(turns))
		n += 1
		if n%1e3 == 0:
			print('processed %ik'%(n/1e3))

	with open(DIR+'/cleaned.txt', 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))


def load_vocab():
	i = 0
	wordtoix = dict()
	ixtoword = dict()
	for word in open(DIR+'/vocab.txt'):
		word = word.strip('\n')
		wordtoix[word] = i
		ixtoword[i] = word
		i += 1
	return ixtoword, wordtoix


def compress():
	import cPickle
	
	x = []
	loadpath = DIR + '/cleaned.txt'
	print('reading '+loadpath)
	with open(loadpath, 'rb') as f:
		for line in f:
			x.append(line.strip('\n').strip())

	n = len(x)
	print('shuffling %i lines'%n)
	ii = list(range(n))
	np.random.seed(9)
	np.random.shuffle(ii)

	print('splitting...')
	sent = []
	ii_picked = []
	for i in ii:
		line = x[i].split('\t')
		if len(line) == TURNS and all([len(z)<MAXLEN for z in line]):
			sent.append(line)
			ii_picked.append(i)
	print('picked %i lines'%len(ii_picked))
 
	def convert_word_to_ix(data):
		result = []
		for conv in data:
			temp_c = []
			for sent in conv:
				temp = []
				for w in sent.split():
					if w in wordtoix:
						temp.append(wordtoix[w])
					else:
						temp.append(3)
				temp.append(2)
				temp_c.append(temp)
			result.append(temp_c)
		return result

	a = int(1e3)
	b = int(2e3)
	val_x = sent[:a]
	test_x = sent[a:b]
	train_x = sent[b:]

	print('writing val')
	with open(DIR + '/val.txt', 'w') as f:
		f.write('\n'.join(['\t'.join(d) for d in val_x]))
	with open(DIR + '/val.line_id', 'w') as f:
		f.write('\n'.join(map(str, ii_picked[:a])))

	print('writing test')
	with open(DIR + '/test.txt', 'w') as f:
		f.write('\n'.join(['\t'.join(d) for d in test_x]))
	with open(DIR + '/test.line_id', 'w') as f:
		f.write('\n'.join(map(str, ii_picked[a:b])))

	print('writing train')
	with open(DIR + '/train.txt', 'w') as f:
		f.write('\n'.join(['\t'.join(d) for d in train_x]))
	with open(DIR + '/train.line_id', 'w') as f:
		f.write('\n'.join(map(str, ii_picked[b:])))

	print('converting to num')
	ixtoword, wordtoix = load_vocab()
	val_x = convert_word_to_ix(val_x)
	test_x = convert_word_to_ix(test_x)
	train_x = convert_word_to_ix(train_x)

	print('dumping...')
	cPickle.dump([train_x, val_x, test_x, wordtoix, ixtoword], open(DIR + "/maluuba.p", "wb"))


MAXLEN = 300
DIR = 'f:/cons_io/data_maluuba'
TURNS = 11

if __name__ == '__main__':
	#preprocess()
	compress()