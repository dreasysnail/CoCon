#coding=utf8

CONTEXT_LEN = 5

if __name__ == '__main__':
	import json
	from tokenize import generate_tokens
	import io
	import random

	dials = []
	# train_file_en = open("data/train.en", "w")
	# train_file_vi = open("data/train.vi", "w")
	# test_file_en = open("data/test.en", "w")
	# test_file_vi = open("data/test.vi", "w")
	# dev_file_en = open("data/dev.en", "w")
	# dev_file_vi = open("data/dev.vi", "w")
	data = open("data/maluuba.txt", 'w')

	for line in open("blis_collected_dialogues_enforced.json"):
		for item in json.loads(line.strip()):
			if 'turns' in item:
				try:
					turns = []
					for turn in item['turns']:
						tokens = []
						if 'text' in turn:
							for tok in generate_tokens(io.StringIO(turn['text']).readline):
								_, t_str, _, _, _ = tok
								tokens.append(t_str)
						turns.append(" ".join(tokens).lower())
					dials.append(turns)
				except:
					pass
	
	# random.shuffle(dials)
	for item in dials:
		print ("\n".join(item) + "\n\n")
	dialouge_num = len(dials)
	print (dialouge_num)
	all_dials = dials
	# test_dials = dials[0:1000]
	# dev_dials = dials[1000:2000]
	# train_dials = dials[2000:]

	for dial in all_dials:
		# for i in range(1, len(dial)):
		data.write("\t".join(dial).replace("\n", " ").replace("\r", " ").strip() + "\n")
			# test_file_en.write(" <s> ".join(dial[max(0, (i-CONTEXT_LEN)):i]).replace("\n", " ").replace("\r", " ").strip() + "\n")
			# test_file_vi.write(dial[i].replace("\n", " ").replace("\r", " ").strip() + "\n")
	# for dial in dev_dials:
	# 	for i in range(1, len(dial)):
	# 		dev_file_en.write(" <s> ".join(dial[max(0, (i-CONTEXT_LEN)):i]).replace("\n", " ").replace("\r", " ").strip() + "\n")
	# 		dev_file_vi.write(dial[i].replace("\n", " ").replace("\r", " ").strip() + "\n")
	# for dial in train_dials:
	# 	for i in range(1, len(dial)):
	# 		train_file_en.write(" <s> ".join(dial[max(0, (i-CONTEXT_LEN)):i]).replace("\n", " ").replace("\r", " ").strip() + "\n")
	# 		train_file_vi.write(dial[i].replace("\n", " ").replace("\r", " ").strip() + "\n")

	# dictionary = {}
	# for dial in train_dials:
	# 	for utt in dial:
	# 		for word in utt.split(" "):
	# 			if word not in dictionary:
	# 				dictionary[word] = 1
	# 			else:
	# 				dictionary[word] += 1
	# #print ("WORD NUM: ", len(dictionary))
	# #for item in sorted(dictionary.items(), key = lambda d:d[1], reverse = True):
	# #	print (item[0] + "\t" + str(item[1]))

	# unk_sen = 0
	# sen = 0
	# for dial in train_dials:
	# 	for utt in dial:
	# 		flist = utt.split(" ")
	# 		index = 0
	# 		while index < len(flist):
	# 			if flist[index] in dictionary and dictionary[flist[index]] <= 3:
	# 				break
	# 			index += 1
	# 		if index < len(flist):
	# 			unk_sen += 1
	# 		sen += 1
	# print (unk_sen, sen)
		
