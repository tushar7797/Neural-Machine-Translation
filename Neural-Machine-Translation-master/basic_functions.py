'''
----------------------------------------------------
FUNCTION DEFINITIONS FOR NEURAL NETWORK ARCHITECTURE
----------------------------------------------------
'''

import tensorflow as tf
from tensorflow.contrib import rnn

import collections
import time
import random
import codecs
import os
import numpy as np

import parameters
import additional_functions as helper_fns

__all__=['embedding','create_or_load_model','load_model','run_full_eval']

def embedding(src_vocab_size,tgt_vocab_size,inembsize) : 

	#print 'Embedding Function'

	embedding_encoder=tf.get_variable(name='embedding_encoder',shape=[src_vocab_size,inembsize],dtype=tf.float32)
	embedding_decoder=tf.get_variable(name='embedding_decoder',shape=[tgt_vocab_size,inembsize],dtype=tf.float32)

	return [embedding_encoder,embedding_decoder]

def load_model(model,ckpt,sess,name) : 
	start_time=time.time()
	model.saver.restore(sess,ckpt)
	sess.run(tf.tables_initializer())
	return model

def create_or_load_model(model,model_dir,sess,name) : 
	latest_ckpt=tf.train.latest_checkpoint(model_dir)
	if latest_ckpt : 
		print 'model obtained from checkpoint'
		model=load_model(model,latest_ckpt,sess,name)
	else : 
		print 'model created'
		start=time.time()
		sess.run(tf.global_variables_initializer())
		sess.run(tf.tables_initializer())

	global_step=model.global_step.eval(session=sess)
	return [model,global_step]

def sample_decode(model,global_step,sess,param,iterator,src_data,tgt_data,iterator_src_placeholder,
	iterator_batch_size_placeholder,summary_writer) : 
	# Pick a random sentence
	decode_id=random.randint(0,len(src_data)-1)

	iterator_feed_dict={iterator_src_placeholder:[src_data[decode_id]],iterator_batch_size_placeholder:1}

	sess.run(iterator.initializer,feed_dict=iterator_feed_dict)

	nmt_output,attention_summary=model.decode(sess)

	translation=helper_fns.translate(nmt_output,
		sent_id=0,tgt_eos='</s>')

	# Printing translation
	print 'Source : ',src_data[decode_id]
	print 'Actual Target : ',tgt_data[decode_id]
	print 'Translation : ',translation


def decode_and_evaluate(model,global_step,sess,param,iterator,
		iterator_feed_dict,ref_file,label) : 
	
	out_dir=param.out_dir

	if global_step==0 : 
		return 0

	print 'Evaluation at global_step=',global_step

	output=trans_file=os.path.join(out_dir,'output_%s.txt'%label)

	start_time=time.time()
	num_sentences=0

	sess.run(iterator.initializer,feed_dict=iterator_feed_dict)

	with codecs.getwriter('utf-8')(tf.gfile.GFile(trans_file,mode='wb')) as trans_f : 
		trans_f.write(' ')

		while True : 
			try : 
				nmt_output,_=model.decode(sess)
				batch_size=nmt_output.shape[0]
				num_sentences+=batch_size

				for i in range(batch_size) : 
					translation=helper_fns.translate(nmt_output[i],sent_id=0,tgt_eos='</s>')
					#print 'Test Translation : ',translation
					#print type(translation)
					trans_f.write((translation+'\n').decode('utf-8'))

			except tf.errors.OutOfRangeError : 
				print num_sentences,' sentences decoded.'
				break

	# BLEU Score
	print 'Reference File : ',ref_file
	print 'Predicted File : ',trans_file


	os.system('python calculate_bleu_score.py '+ref_file+' '+trans_file)















