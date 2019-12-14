'''
----------------------------
TRAIN AND EVALUATE THE MODEL
----------------------------
'''

import collections
import os
import time

import tensorflow as tf

import basic_functions
import model_attention
import data_preparation
import additional_functions

__all__=['train']

class TrainModel(collections.namedtuple('TrainModel',('graph','model','iterator'))) :  
	pass

def create_train_model(param,scope) : 
	
	graph=tf.Graph()

	with graph.as_default(), tf.container(scope or 'train') : 
	
		# Getting vocabulary
		vocab=data_preparation.vocab_lookup_tables()

		# Iterator for training
		batch_size=param.batch_size
		src_dataset=tf.data.TextLineDataset('Datasets/train.en')
		tgt_dataset=tf.data.TextLineDataset('Datasets/train.vi')

		train_iterator=data_preparation.train_dataset(src_dataset,tgt_dataset,
			batch_size,vocab.src_eos_id,vocab.tgt_eos_id,
			vocab.tgt_sos_id,vocab.src_vocab_table,vocab.tgt_vocab_table,
			src_max_len=param.src_max_len,tgt_max_len=param.tgt_max_len)
		#print train_iterator

		# Building the model
		model=model_attention.AttentionModel(param=param,
			mode=tf.contrib.learn.ModeKeys.TRAIN,iterator=train_iterator,vocab=vocab,scope=scope)

	return TrainModel(graph=graph,model=model,iterator=train_iterator)


class InferModel(collections.namedtuple('InferModel',('graph','model','src_placeholder',
	'batch_size_placeholder','iterator'))) : 
	pass

def create_infer_model(param,scope) : 

	graph=tf.Graph()

	with graph.as_default(),tf.container(scope or 'infer') : 

		# Getting Vocabulary
		vocab=data_preparation.vocab_lookup_tables()

		# Placeholders
		src_placeholder=tf.placeholder(shape=[None],dtype=tf.string)
		batch_size_placeholder=tf.placeholder(shape=[],dtype=tf.int64)

		src_dataset=tf.data.Dataset.from_tensor_slices(src_placeholder)

		# Iterator
		infer_iterator=data_preparation.test_dataset(src_dataset=src_dataset,
			batch_size=batch_size_placeholder,src_eos_id=vocab.src_eos_id,
			src_vocab_table=vocab.src_vocab_table,src_max_len=param.src_max_len_inf)

		model=model_attention.AttentionModel(param=param,
			mode=tf.contrib.learn.ModeKeys.INFER,iterator=infer_iterator,vocab=vocab,scope=scope)

	return InferModel(graph=graph,model=model,src_placeholder=src_placeholder,
		batch_size_placeholder=batch_size_placeholder,iterator=infer_iterator)

def evaluation(infer_model,infer_sess,model_dir,param,global_step,val_or_test) : # Evaluate Test Data
	
	if val_or_test=='val' : # Validation
		print 'Evaluating on validation data'
		test_src_file='Datasets/tst2013.en'
		test_tgt_file='Datasets/tst2013.vi'
	elif val_or_test=='test' : # Test
		print 'Evaluaring on test data'
		test_src_file='Datasets/tst2012.en'
		test_tgt_file='Datasets/tst2012.vi'
	else : 
		raise ValueError('Give appropriate labels.')


	with infer_model.graph.as_default() : 
		loaded_infer_model,global_step_temp=basic_functions.create_or_load_model(
			infer_model.model,model_dir,infer_sess,'infer')

	test_infer_iterator_feed_dict={infer_model.src_placeholder:additional_functions.load_data(test_src_file),
		infer_model.batch_size_placeholder:param.infer_batch_size}

	basic_functions.decode_and_evaluate(model=loaded_infer_model,global_step=global_step,
		sess=infer_sess,param=param,iterator=infer_model.iterator,
		iterator_feed_dict=test_infer_iterator_feed_dict,ref_file=test_tgt_file,label=val_or_test)



def train(param,target_session='',scope=None) : 
	
	out_dir=param.out_dir
	num_train_steps=param.num_train_steps
	steps_per_stats=param.steps_per_stats

	train_model=create_train_model(param=param,scope=scope)
	infer_model=create_infer_model(param=param,scope=scope)

	# Loading data for sample decoding
	dev_src_file='Datasets/tst2013.en'
	dev_tgt_file='Datasets/tst2013.vi'
	sample_src_data=additional_functions.load_data(dev_src_file)
	sample_tgt_data=additional_functions.load_data(dev_tgt_file)

	summary_name='train_log'
	model_dir=param.out_dir

	# Log and output files
	log_file=os.path.join(out_dir,'log_%d'%time.time())
	log_f=tf.gfile.GFile(log_file,mode='a')

	# Sessions
	train_sess=tf.Session(target=target_session,graph=train_model.graph)
	infer_sess=tf.Session(target=target_session,graph=infer_model.graph)

	with train_model.graph.as_default() : 
		#create or load model
		loaded_train_model,global_step=basic_functions.create_or_load_model(
			train_model.model,model_dir,train_sess,'train')

	summary_writer=tf.summary.FileWriter(os.path.join(out_dir,summary_name),train_model.graph)


	train_sess.run(train_model.iterator.initializer)

	while global_step<num_train_steps : 

		if global_step%steps_per_stats==0 : 
			print 'Global Step : ',global_step,', Epoch Step : ',param.epoch_step

		start_time=time.time()
		try : 

			step_result=loaded_train_model.train(train_sess)			
			param.epoch_step+=1

		except tf.errors.OutOfRangeError : 
			print 'Gone through the dataset once. Starting the next epoch..'
			# Finished this run through the training set
			# Go to next epoch
			param.epoch_step=0
			train_sess.run(train_model.iterator.initializer)

			#Decode a random sentence from source data
			print 'After completion of epoch : '
			print 'Step Result : ',step_result
			evaluation(infer_model,infer_sess,model_dir,param,global_step,'val')
			
			continue
		
		(_,step_loss,step_predict_count,global_step,step_word_count,
			batch_size,step_grad_norm,learning_rate)=step_result

		if global_step%steps_per_stats==0 :  
			print 'Loss : ',step_loss,'  ,Learning Rate :  ',learning_rate

		if global_step%50==0 : 
			print 'Saving checkpoint...'
			loaded_train_model.saver.save(train_sess,os.path.join(out_dir,'translate.ckpt'),
				global_step=global_step)

		if global_step%200==0 : 
			#print step_result

			with infer_model.graph.as_default() : 
				loaded_infer_model,global_step_temp=basic_functions.create_or_load_model(
					infer_model.model,model_dir,infer_sess,'infer')

			print 'Sample Decode'
			basic_functions.sample_decode(loaded_infer_model,global_step,infer_sess,
				param,infer_model.iterator,sample_src_data,sample_tgt_data,
				infer_model.src_placeholder,infer_model.batch_size_placeholder,summary_writer)

		if global_step%200==0 : 
			evaluation(infer_model,infer_sess,model_dir,param,global_step,'val')
			


	print 'Training done, evaluating now..'
	# Testing on the test set now
	evaluation(infer_model,infer_sess,model_dir,param,global_step,'test')










