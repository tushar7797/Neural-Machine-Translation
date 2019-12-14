'''
--------------------
CALCULATE BLEU SCORE
--------------------
'''

from nltk.translate.bleu_score import sentence_bleu
import sys

ref_file=open(sys.argv[1],'r').readlines()
pred_file=open(sys.argv[2],'r').readlines()

ref_len=len(ref_file)
pred_len=len(pred_file)

assert ref_len==pred_len

bleu=0
for i in range(ref_len) : 
	bleu+=sentence_bleu([ref_file[i].strip().split()],pred_file[i].strip().split())

print '\nBLEU Score : ',bleu/float(ref_len)