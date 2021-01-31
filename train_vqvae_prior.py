from IPython import embed
from dienen.core.file import load_model
import dienen
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

#Turn weights and config into dienen model:

#config_path = 'models/cnn_vqvae_speech.yaml'
#weights_path = 'pretrained_models/vqvae-librispeech/weights.03-0.02.hdf5'
#vqvae_model = dienen.Model(config_path)
#vqvae_model.build()
#vqvae_model.core_model.model.load_weights(weights_path)
#vqvae_model.save_model('pretrained_models/vqvae-librispeech/model.dnn')

#loaded_model = load_model('pretrained_models/vqvae-librispeech/model.dnn')
#loaded_model.core_model.model.summary()
#embed()

#Make training data for prior model:
#vqvae_encoder = dienen.Model('models/cnn_vqvae_speech_encode.yaml')
#vqvae_encoder.build()

#import librispeech
#audio_train_data, audio_test_data = librispeech.read_librispeech('../LibriSpeech')

#audio_train_indexs = vqvae_encoder.core_model.model.predict(audio_train_data)

#Train prior:
#train_data = joblib.load('ls-codes')

#class train_generator(tf.keras.utils.Sequence):
#    def __init__(self,data,batch_size):
#        self.data = data
#        self.index = np.arange(0,len(data))
#        self.batch_size = batch_size

#    def __getitem__(self,i):
#        batch_idxs = np.take(self.index,np.arange(i*self.batch_size,(i+1)*self.batch_size))
#        batch_data = self.data[batch_idxs] + 1
#        oh_data = tf.one_hot(batch_data,depth=513,axis=-1)
#        x = tf.roll(oh_data,1,axis=1)
#        x = x.numpy()
#        x[:,0,:,:] = 0
        
#        return x, oh_data.numpy()
        
#    def on_epoch_end(self):
#        self.index = np.random.permutation(self.index)

#    def __len__(self):
#        return len(self.data)//self.batch_size

#gen = train_generator(train_data,batch_size = 64)

#prior_model = dienen.Model('models/cnn_vqvae_speech_prior.yaml')
#prior_model.build()
#prior_model.core_model.model.compile(loss='categorical_crossentropy',optimizer='adam')
#cbks = [tf.keras.callbacks.ModelCheckpoint('../ckpts/prior/weights.{epoch:02d}-{loss:.2f}.hdf5')]
#prior_model.core_model.model.fit(gen,epochs=20,callbacks=cbks)
#prior_model.save_model('pretrained_models/vqvae-librispeech/prior.dnn')
#embed()

train_data = joblib.load('ls-codes')
random_start = np.zeros((1,8,8,513))
random_start[0,1,:] = tf.one_hot(train_data[0][0,:] + 1,depth=513,axis=-1).numpy() #Elijo como primer sample uno existente

prior_model = dienen.Model('models/cnn_vqvae_speech_prior.yaml')
prior_model.build()
prior_model.core_model.model.load_weights('../ckpts/prior/weights.20-0.70.hdf5')
for i in range(random_start.shape[1]-2):
    predicted = prior_model.core_model.model.predict(random_start)
    random_start[0,i+2,:,:] = predicted[0,i+1,:,:]

categorical_distribution = tfp.distributions.Categorical(logits=random_start)
generated_code = categorical_distribution.sample().numpy()
generated_code -= 1
generated_code[generated_code==-1]=0

decoder_model = dienen.Model('models/cnn_vqvae_speech_decode.yaml')
decoder_model.build()

embed()