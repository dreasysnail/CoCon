# -*- coding: utf-8 -*-
# Xiang Gao

import os, sys, io, re
import numpy as np
from nltk.tokenize import TweetTokenizer

def norm_sentence(txt):
	txt = txt.lower()

	# return None if quoted
	if "__mention__:" in txt.replace(' ',''):	# used to have bug
		return None

	# remove illegal char but keep __mention__
	txt = txt.replace('__mention__','MENTION')
	txt = re.sub(r"[^A-Za-z0-9():,\.!?' ]", "", txt)
	txt = txt.replace('MENTION','__mention__')	

	# url 
	words = []
	for word in txt.lower().split():
		i = word.find('http') 
		if i >= 0:
			word = word[:i] + ' ' + '__url__'
		words.append(word.strip())
	txt = ' '.join(words)

	# contraction
	add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
	tokenizer = TweetTokenizer(preserve_case=False)
	txt = ' '.join(tokenizer.tokenize(txt)) + ' '
	txt = txt.replace("won't", "will n't")
	txt = txt.replace("can't", "can n't")
	for a in add_space:
		txt = txt.replace(a+' ', " "+a+' ')
	
	# irregular words
	mapping = {
		''
	}

	# remove un-necessary space
	return ' '.join(txt.split())




if __name__ == '__main__':

	ss = [
		" I don't know this shit!!!how about this?http://adasdas.com",
		"sure:)",
		'I love emoji😂😂😂😂',
		"“__MENTION__: “__MENTION__: I want some taco's.”I'm eating tacos tonight,😝!!” Ugh I'm jealous😞 yo ass can share!",
		'have you asked __MENTION__?',
		"sure i'll",
		]
	for s in ss:
		print(norm_sentence(s))

	#extract_head('d:/word2vec/glove.twitter.27B/glove.twitter.27B.200d.txt')