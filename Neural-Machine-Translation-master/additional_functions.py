'''
--------------------
ADDITIONAL FUNCTIONS
--------------------
'''


import codecs
import collections

import tensorflow as tf

__all__=['load_data','translate','format_text']

def load_data(data_file) : 
	
	with codecs.getreader('utf-8')(tf.gfile.GFile(data_file,mode='r')) as f : 
		infer_data=f.read().splitlines()

	return infer_data

def format_text(words) : 

	# Sequence of words into a sentence
	#if(not hasattr(words,'__len__')) and not isinstance(words,collections.Iterable) : 
	#	words=[words]

	return ' '.join(words)



def translate(nmt_output,sent_id,tgt_eos) : 

	if tgt_eos : 
		tgt_eos=tgt_eos.encode('utf-8')
	
	output=list(nmt_output)

	if '</s>' in output : # consider only uptil first end-of-sentence character
		eos_index=output.index('</s>')
		output=output[0:eos_index]

	translation=format_text(output)

	return translation
