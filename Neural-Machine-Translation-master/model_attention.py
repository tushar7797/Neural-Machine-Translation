'''
---------
ATTENTION
---------
'''

import collections

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core

import basic_functions as fn

__all__=['AttentionModel']

class AttentionModel() : 
	
	def __init__(self,param,mode,iterator,vocab,scope) : 
		
		self.iterator=iterator
		#print self.iterator
		self.mode=mode
		self.scope=scope
		
		self.src_vocab_table=vocab.src_vocab_table
		self.tgt_vocab_table=vocab.tgt_vocab_table
		self.src_vocab_size=vocab.src_vocab_size
		self.tgt_vocab_size=vocab.tgt_vocab_size

		self.tgt_sos_id=vocab.tgt_sos_id
		self.tgt_eos_id=vocab.tgt_eos_id

		self.num_layers_encoder=param.num_layers_encoder
		self.num_layers_decoder=param.num_layers_decoder
		self.dropout=param.dropout

		self.time_major=param.time_major

		# Setting initializer
		if param.init_method=='uniform' : 
			initializer=tf.random_uniform_initializer(-param.init_weight,param.init_weight)
			tf.get_variable_scope().set_initializer(initializer)
		elif param.init_method=='gaussian' : 
			initializer=tf.truncated_normal_initializer(mean=param.init_mean,
				stddev=param.init_std,seed=None,dtype=tf.float32)
			tf.get_variable_scope().set_initializer(initializer)
		elif param.init_method=='xavier' : 
			initializer=tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
			tf.get_variable_scope().set_initializer(initializer)
		else : 
			raise ValueError('Give Valid method of weight initialisation')

		# Embedding
		[self.embedding_encoder,self.embedding_decoder]=fn.embedding(self.src_vocab_size,
			self.tgt_vocab_size,param.inembsize)

		self.batch_size=tf.size(self.iterator.source_seq_len)

		# Projection Layer # Logits
		with tf.variable_scope(scope or 'build_network') : 
			with tf.variable_scope('decoder/output_projection') : 
				self.output_layer=layers_core.Dense(units=self.tgt_vocab_size,use_bias=False,name='output_projection')

		# Results
		results=self.build_model(param,scope)

		if self.mode==tf.contrib.learn.ModeKeys.TRAIN : 			
			self.train_loss=results[1]
			self.word_count=tf.reduce_sum(self.iterator.source_seq_len)+tf.reduce_sum(self.iterator.target_seq_len)
		elif self.mode==tf.contrib.learn.ModeKeys.INFER : 
			self.infer_logits,_,self.final_context_state,self.sample_id=results
			self.sample_words=vocab.reverse_tgt_vocab_table.lookup(tf.to_int64(self.sample_id))
		else : 
			raise ValueError('Give a valid mode')

		if self.mode!=tf.contrib.learn.ModeKeys.INFER : 
			self.predict_count=tf.reduce_sum(self.iterator.target_seq_len)

		self.global_step=tf.Variable(0,trainable=False)

		parameters=tf.trainable_variables()

		if self.mode==tf.contrib.learn.ModeKeys.TRAIN : 
			self.learning_rate=tf.constant(param.learning_rate)
			self.learning_rate=self.decay_learning_rate(param)

			# Adam Optimizer
			opt=tf.train.AdamOptimizer(self.learning_rate)

			# Gradients
			gradients=tf.gradients(self.train_loss,parameters,colocate_gradients_with_ops=True)
			clipped_gradient,gradient_norm=tf.clip_by_global_norm(gradients,param.max_grad_norm)
			gradient_norm_summary=[tf.summary.scalar('grad_norm',gradient_norm)]
			gradient_norm_summary.append(tf.summary.scalar('clipped_gradient',tf.global_norm(clipped_gradient)))
			self.grad_norm=gradient_norm

			self.update=opt.apply_gradients(zip(clipped_gradient,parameters),global_step=self.global_step)

			# summary ND now

		if self.mode==tf.contrib.learn.ModeKeys.INFER : 
			self.infer_summary=self.create_attention_images_summary(self.final_context_state)

		self.saver=tf.train.Saver(tf.global_variables(),max_to_keep=param.num_ckpts)

	def decay_learning_rate(self,param) : 

		return tf.cond(self.global_step<param.start_decay_global_step,
			lambda : self.learning_rate,
			lambda : tf.train.exponential_decay(self.learning_rate,
				(self.global_step-param.start_decay_global_step),param.decay_steps,
				param.decay_factor,staircase=True),name='learning_rate_decay_cond')


	def make_single_lstm_unit(self,num_units) : 

		#print 'Making a single LSTM unit'

		single_cell=rnn.BasicLSTMCell(num_units,activation=tf.tanh,forget_bias=1.0)
		
		if self.mode==tf.contrib.learn.ModeKeys.TRAIN : 
			dropout=self.dropout
		else : 
			dropout=1.0

		#print 'Checking, mode : ',self.mode,', dropout : ',dropout

		if dropout==1.0 : 
			pass
		elif dropout>0.0 and dropout<1.0 : 
			single_cell=rnn.DropoutWrapper(cell=single_cell,input_keep_prob=dropout)
		else : 
			print 'Invalid dropout'

		return single_cell

	def make_lstm_cell_list(self,param,num_layers) : 

		num_units=param.num_lstm_units
		cell_list=[]
		for i in range(num_layers) : 
			single_cell=self.make_single_lstm_unit(num_units)
			cell_list.append(single_cell)

		if num_layers==1 : 
			return cell_list[0]
		else : 
			return rnn.MultiRNNCell(cell_list)

		
	def make_encoder(self,param)  : 
	
		#print 'Making the encoder'

		iterator=self.iterator

		num_units=param.num_lstm_units
		num_layers=self.num_layers_encoder

		source=iterator.source

		if self.time_major : 
			source=tf.transpose(source) # unsure

		with tf.variable_scope('encoder') as scope : 
			# Look up embedding
			encoder_emb_inp=tf.nn.embedding_lookup(self.embedding_encoder,source)
			#print 'encoder emb inp',encoder_emb_inp

			# Bidirectional LSTM
			# Encoder Cell
			num_bi_layers=int(num_layers/2.0)
			encoder_cell_fw=self.make_lstm_cell_list(param,num_bi_layers)
			encoder_cell_bw=self.make_lstm_cell_list(param,num_bi_layers)

			encoder_bidirn_output,encoder_bidirn_state=tf.nn.bidirectional_dynamic_rnn(time_major=self.time_major,
				dtype=tf.float32,cell_fw=encoder_cell_fw,cell_bw=encoder_cell_bw,inputs=encoder_emb_inp,
				sequence_length=self.iterator.source_seq_len)

			encoder_output=tf.concat(encoder_bidirn_output,-1)

			if num_bi_layers==1 : # check why
				encoder_state=encoder_bidirn_state
			else : 
				encoder_state=[]
				for i in range(num_bi_layers) : 
					encoder_state.append(encoder_bidirn_state[0][i]) # forward
					encoder_state.append(encoder_bidirn_state[1][i]) # backward 
				encoder_state=tuple(encoder_state)

		return [encoder_output,encoder_state]

	def attention_mechanism_fn(self,attention_option,num_units,memory,source_seq_len) : 
	
		# Note : Mode not used here
		if attention_option=='luong' : 
			attention_mechanism=tf.contrib.seq2seq.LuongAttention(num_units=num_units,
				memory=memory,memory_sequence_length=source_seq_len)

		elif attention_option=='scaled_luong' : 
			attention_mechanism=tf.contrib.seq2seq.LuongAttention(num_units=num_units,
				memory=memory,memory_sequence_length=source_seq_len,scale=True)

		elif attention_option=='bahdanau' : 
			attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(num_units=num_units,
				memory=memory,memory_sequence_length=source_seq_len)

		elif attention_option=='normed_bahdanau' : 
			attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(num_units=num_units,
				memory=memory,memory_sequence_length=source_seq_len,normalize=True)

		else : 
			raise ValueError('Unknown attention option : %s',attention_option)

		return attention_mechanism



	def make_decoder_cell(self,param,encoder_output,encoder_state,source_seq_len) : 
		
		attention_option=param.attention_option
		num_units=param.num_lstm_units
		num_layers=self.num_layers_decoder

		if self.time_major : 
			memory=tf.transpose(encoder_output,[1,0,2])
		else : 
			memory=encoder_output

		batch_size=self.batch_size

		print 'MEMORY : ',memory
		# Attention Mechanism
		attention_mechanism=self.attention_mechanism_fn(attention_option,
			num_units,memory,source_seq_len)

		cell=self.make_lstm_cell_list(param,num_layers)

		# Generate alignment history for test mode alone
		alignment_history=(self.mode==tf.contrib.learn.ModeKeys.INFER)

		# Attention Wrapper
		cell=tf.contrib.seq2seq.AttentionWrapper(cell=cell,attention_mechanism=attention_mechanism,
			attention_layer_size=num_units,alignment_history=alignment_history,
			output_attention=param.output_attention,name='attention')

		if param.pass_hidden_state : 
		
			W_hidden=tf.get_variable(name='W_hidden',shape=[2,num_units,num_units],
				initializer=tf.contrib.layers.xavier_initializer())
			#encoder_final_state_c = tf.concat((encoder_state[0].c, encoder_state[1].c), 1)
			decoder_init_state_c_fw = tf.matmul(encoder_state[0].c,W_hidden[0])
			decoder_init_state_c_fw=tf.tanh(decoder_init_state_c_fw)
			decoder_init_state_c_bw = tf.matmul(encoder_state[1].c,W_hidden[0])
			decoder_init_state_c_bw=tf.tanh(decoder_init_state_c_bw)

			#encoder_final_state_h = tf.concat((encoder_state[0].h, encoder_state[1].h), 1)
			decoder_init_state_h_fw= tf.matmul(encoder_state[0].h,W_hidden[1])
			decoder_init_state_h_fw=tf.tanh(decoder_init_state_h_fw)
			decoder_init_state_h_bw= tf.matmul(encoder_state[1].h,W_hidden[1])
			decoder_init_state_h_bw=tf.tanh(decoder_init_state_h_bw)

			decoder_initial_state_fw= tf.contrib.rnn.LSTMStateTuple(
    			c=decoder_init_state_c_fw,
    			h=decoder_init_state_h_fw)
			decoder_initial_state_bw= tf.contrib.rnn.LSTMStateTuple(
    			c=decoder_init_state_c_bw,
    			h=decoder_init_state_h_bw)

			decoder_initial_state=(decoder_initial_state_fw,decoder_initial_state_bw)

			#print encoder_state,'\n\n',decoder_initial_state,'\n\n'
			decoder_init_state=cell.zero_state(batch_size,dtype=tf.float32).clone(cell_state=decoder_initial_state)
			#decoder_init_state=cell.zero_state(batch_size,dtype=tf.float32).clone(cell_state=encoder_state)
			
		else : 
			decoder_init_state=cell.zero_state(batch_size,dtype=tf.float32)

		return [cell,decoder_init_state]



	def make_decoder(self,param,encoder_output,encoder_state) : 
		
		tgt_sos_id=self.tgt_sos_id
		tgt_eos_id=self.tgt_eos_id

		iterator=self.iterator

		# For inference
		if param.src_max_len_inf : 
			max_iter=param.src_max_len_inf
		else : 
			max_enc_len=tf.reduce_max(iterator.source_seq_len)
			max_iter=tf.to_int32(tf.round(tf.to_float(max_enc_len)*2.0))

		with tf.variable_scope('decoder') as decoder_scope : 
			# Make the decoder
			# Decoder Cell
			[decoder_cell,decoder_init_state]=self.make_decoder_cell(param,
				encoder_output,encoder_state,iterator.source_seq_len)

		#print decoder_init_state

			if self.mode!=tf.contrib.learn.ModeKeys.INFER : # Train 

				target_input=iterator.target_input

				if self.time_major : 
					target_input=tf.transpose(target_input)
				#print target_input

				decoder_emb_inp=tf.nn.embedding_lookup(self.embedding_decoder,target_input)
				#print decoder_emb_inp

				# Helper
				helper=tf.contrib.seq2seq.TrainingHelper(inputs=decoder_emb_inp,
					sequence_length=iterator.target_seq_len,time_major=self.time_major)

				# Decoder
				decoder_used=tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
					helper=helper,initial_state=decoder_init_state)

				# Dynamic Decoding
				decoder_output,final_context_state,_=tf.contrib.seq2seq.dynamic_decode(decoder=decoder_used,
					output_time_major=self.time_major,swap_memory=True,scope=decoder_scope)

				sample_id=decoder_output.sample_id
				logits=self.output_layer(decoder_output.rnn_output)

			else : # Inference
				
				start_tokens=tf.fill([self.batch_size],tgt_sos_id)
				end_token=tgt_eos_id

				# No beam search

				# Helper
				sampling_temp=param.sampling_temp

				if sampling_temp>0.0 : # sample from prev logits to get decoder's input
					helper=tf.contrib.seq2seq.SampleEmbeddingHelper(embedding=self.embedding_decoder,
						start_tokens=start_tokens,end_token=end_token,
						softmax_temperature=sampling_temp)

				else : # most probable word from prev logits to get decoder's input
					helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_decoder,
						start_tokens=start_tokens,end_token=end_token)

				# Decoder
				decoder_used=tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
					helper=helper,initial_state=decoder_init_state,output_layer=self.output_layer)

				# Dynamic Decoding
				decoder_output,final_context_state,_=tf.contrib.seq2seq.dynamic_decode(decoder=decoder_used,
					maximum_iterations=max_iter,
					output_time_major=self.time_major,swap_memory=True,scope=decoder_scope)

				sample_id=decoder_output.sample_id
				logits=decoder_output.rnn_output

		return [logits,sample_id,final_context_state]

	def compute_loss(self,logits) : 

		target_output=self.iterator.target_output

		if self.time_major : 
			target_output=tf.transpose(target_output)

		time_axis=0 if self.time_major else 1
		max_time=tf.shape(target_output)[time_axis]

		cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output,logits=logits)

		target_weights=tf.sequence_mask(lengths=self.iterator.target_seq_len,maxlen=max_time,dtype=logits.dtype)
		if self.time_major : 
			target_weights=tf.transpose(target_weights)

		loss=tf.reduce_sum(cross_entropy*target_weights)/tf.to_float(self.batch_size)

		return loss

	def build_model(self,param,scope) : 
		
		
		with tf.variable_scope(scope or 'dynamic_seq2seq',dtype=tf.float32) : 
			# Encoder
			[encoder_output,encoder_state]=self.make_encoder(param)

			# Decoder
			[logits,sample_id,final_context_state]=self.make_decoder(param,encoder_output,encoder_state)

			# Loss
			if self.mode!=tf.contrib.learn.ModeKeys.INFER : 
				loss=self.compute_loss(logits)
			else : 
				loss=None


		return [logits,loss,final_context_state,sample_id]



	def create_attention_images_summary(self,final_context_state) : 
		# Attention Images
		attention_images=(final_context_state.alignment_history.stack())
		attention_images=tf.expand_dims(tf.transpose(attention_images,[1,2,0]),-1)
		attention_images*=255 #scaling
		attention_summary=tf.summary.image('attention_images',attention_images)
		return attention_summary

	def train(self,sess) : 
		assert self.mode==tf.contrib.learn.ModeKeys.TRAIN
		return sess.run([self.update,self.train_loss,self.predict_count,self.global_step,
			self.word_count,self.batch_size,self.grad_norm,self.learning_rate])		

	def eval(self,sess) : 
		assert self.mode==tf.contrib.learn.ModeKeys.EVAL
		return sess.run([self.eval_loss,self.predict_count,self.batch_size])

	def infer(self,sess) : 
		assert self.mode==tf.contrib.learn.ModeKeys.INFER
		return sess.run([self.infer_logits,self.infer_summary,self.sample_id,self.sample_words])

	def decode(self,sess) : 
		_,infer_summary,_,sample_words=self.infer(sess)

		if self.time_major : 
			sample_words=sample_words.transpose()
		return sample_words,infer_summary




	






