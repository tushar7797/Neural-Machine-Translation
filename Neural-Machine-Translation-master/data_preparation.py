'''
---------------------------------
DATA EXTRACTION AND PREPROCESSING
---------------------------------
'''

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import collections

__all__=['Vocab','Data','vocab_lookup_tables','train_dataset','test_dataset']

class Vocab(collections.namedtuple('Vocab',('src_vocab_table','tgt_vocab_table','reverse_tgt_vocab_table',
	'src_eos_id','tgt_eos_id','tgt_sos_id','src_vocab_size','tgt_vocab_size'))) : 
	pass

class Data(collections.namedtuple('Data',('dataset','initializer','source',
	'target_input','target_output','source_seq_len','target_seq_len'))) : 
	pass

def vocab_lookup_tables() : 

	#print 'Loading and collecting data....'

	#opening required files
	english_file=open('Vocabulary_Files/vocab.en','r')
	viet_file=open('Vocabulary_Files/vocab.vi','r')

	#preprocessing and counting vocablary
	#source : english, target : vietnamese
	english_vocab=[line.rstrip('\n') for line in english_file]
	viet_vocab=[line.rstrip('\n') for line in viet_file]

	UNK='<unk>'
	SOS='<s>'
	EOS='</s>'

	eng_vocab_size=len(english_vocab)
	viet_vocab_size=len(viet_vocab)
	#print 'English Vocabulary Size : ',eng_vocab_size
	#print 'Vietnamese Vocabulary Size : ',viet_vocab_size

	src_vocab_size=eng_vocab_size
	tgt_vocab_size=viet_vocab_size
	src_max_len=50
	tgt_max_len=50

	#creating vocab lookup tables
	src_vocab_table=tf.contrib.lookup.index_table_from_file(vocabulary_file='Vocabulary_Files/vocab.en',vocab_size=src_vocab_size,default_value=0)
	tgt_vocab_table=tf.contrib.lookup.index_table_from_file(vocabulary_file='Vocabulary_Files/vocab.vi',vocab_size=tgt_vocab_size,default_value=0)
	reverse_tgt_vocab_table=lookup_ops.index_to_string_table_from_file('Vocabulary_Files/vocab.vi',default_value=UNK)

	src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(EOS)), tf.int32)
	tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(SOS)), tf.int32)
	tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(EOS)), tf.int32)

	return Vocab(src_vocab_table=src_vocab_table,
		tgt_vocab_table=tgt_vocab_table,
		reverse_tgt_vocab_table=reverse_tgt_vocab_table,
		src_eos_id=src_eos_id,
		tgt_eos_id=tgt_eos_id,
		tgt_sos_id=tgt_sos_id,
		src_vocab_size=src_vocab_size,
		tgt_vocab_size=tgt_vocab_size)

def batching_function_train(x,batch_size,src_eos_id,tgt_eos_id) : 
	return x.padded_batch(batch_size=batch_size,padded_shapes=([-1],[-1],[-1],[],[]),padding_values=(src_eos_id,tgt_eos_id,tgt_eos_id,0,0))


def batching_function_test(x,batch_size,src_eos_id) : 
	return x.padded_batch(batch_size=batch_size,padded_shapes=([-1],[]),padding_values=(src_eos_id,0))



def train_dataset(src_dataset,tgt_dataset,batch_size,src_eos_id,tgt_eos_id,tgt_sos_id,
	src_vocab_table,tgt_vocab_table,src_max_len,tgt_max_len) : 


	UNK='<unk>'
	SOS='<s>'
	EOS='</s>'



	#combining source and target datasets
	src_tgt_dataset=tf.data.Dataset.zip((src_dataset,tgt_dataset))
	num_shards=1 # increase if using distributed training
	src_tgt_dataset=src_tgt_dataset.shard(num_shards=num_shards,index=0)

	#shuffling
	output_buffer_size=batch_size*1000 # huge buffer size to ensure shuffling 
	src_tgt_dataset=src_tgt_dataset.shuffle(buffer_size=output_buffer_size,seed=None,reshuffle_each_iteration=True)
	src_tgt_dataset=src_tgt_dataset.map(lambda src,tgt : (tf.string_split([src]).values,tf.string_split([tgt]).values),num_parallel_calls=4).prefetch(output_buffer_size)

	#filter zero length input sequences
	src_tgt_dataset=src_tgt_dataset.filter(lambda src,tgt : tf.logical_and(tf.size(src)>0,tf.size(tgt)>0))

	#convert words into their integer ids using previously generated lookup table
	src_tgt_dataset=src_tgt_dataset.map(lambda src,tgt : (tf.cast(src_vocab_table.lookup(src),tf.int32),tf.cast(tgt_vocab_table.lookup(tgt),tf.int32)),num_parallel_calls=4).prefetch(output_buffer_size)

	#append sos and eos to beginning and end of target input respectively
	src_tgt_dataset=src_tgt_dataset.map(lambda src,tgt : (src,tf.concat(([tgt_sos_id],tgt),0),tf.concat((tgt,[tgt_eos_id]),0)),num_parallel_calls=4).prefetch(output_buffer_size)

	#add sequence lengths
	src_tgt_dataset=src_tgt_dataset.map(lambda src,tgt_in,tgt_out : (src,tgt_in,tgt_out,tf.size(src),tf.size(tgt_in)),num_parallel_calls=4).prefetch(output_buffer_size)

	#buckets (to group by sequence lengths) not using now

	batched_dataset=batching_function_train(src_tgt_dataset,batch_size,src_eos_id,tgt_eos_id)
	batched_iter=batched_dataset.make_initializable_iterator()
	#to get next batch
	(src_ids,tgt_input_ids,tgt_output_ids,src_seq_len,tgt_seq_len)=(batched_iter.get_next())
	batch_initializer=batched_iter.initializer

	return Data(dataset=src_tgt_dataset,
		initializer=batch_initializer,
		source=src_ids,
		target_input=tgt_input_ids,
		target_output=tgt_output_ids,
		source_seq_len=src_seq_len,
		target_seq_len=tgt_seq_len)




def test_dataset(src_dataset,batch_size,src_eos_id,src_vocab_table,src_max_len) : 


	UNK='<unk>'
	SOS='<s>'
	EOS='</s>'


	# Splitting sentence into words
	src_dataset=src_dataset.map(lambda src : tf.string_split([src]).values)
	if src_max_len : 
		src_dataset=src_dataset.map(lambda src : src[:src_max_len])

	# Convert words into their integer ids using the previously generated lookup tables
	src_dataset=src_dataset.map(lambda src : tf.cast(src_vocab_table.lookup(src),tf.int32))

	# Adding in the word counts
	src_dataset=src_dataset.map(lambda src : (src,tf.size(src)))
	batched_dataset=batching_function_test(src_dataset,batch_size,src_eos_id)
	batched_iter=batched_dataset.make_initializable_iterator()
	#to get next batch
	(src_ids,src_seq_len)=batched_iter.get_next()
	batch_initializer=batched_iter.initializer

	return Data(dataset=src_dataset,
		initializer=batch_initializer,
		source=src_ids,
		target_input=None,
		target_output=None,
		source_seq_len=src_seq_len,
		target_seq_len=None)



