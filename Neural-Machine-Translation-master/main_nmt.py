'''
--------------------------------------------------------------
NEURAL MACHINE TRANSLATION : ENGLISH TO VIETNAMESE TRANSLATION
--------------------------------------------------------------
'''

import tensorflow as tf

import parameters
import train_nmt
##import inference

def nmt_start(unused_argv) : 

	param=parameters.create_params_hparams()
	train_fn=train_nmt.train 

	#output directory
	out_dir=param.out_dir
	if not tf.gfile.Exists(out_dir) : 
		tf.gfile.MakeDirs(out_dir)

	if not param.ckpt : 
		ckpt=tf.train.latest_checkpoint(out_dir)

	target_session=''

	# to be done
	# Separate code file for Evaluation

	# else call train function
	train_fn(param,target_session=target_session)




if __name__=='__main__' : 
	tf.app.run(main=nmt_start)