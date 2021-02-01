from IPython import embed
from dienen.core.file import load_model
import dienen
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import librispeech

#Turn weights and config into dienen model:

#config_path = 'models/cnn_vqvae_speech.yaml'
#weights_path = 'pretrained_models/vqvae-librispeech/weights.20-0.08.hdf5'
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
#joblib.dump(audio_train_indexs,'ls-codes')
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
#prior_model.core_model.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='categorical_accuracy')
#cbks = [tf.keras.callbacks.ModelCheckpoint('../ckpts/prior/weights.{epoch:02d}-{loss:.2f}.hdf5')]
#prior_model.core_model.model.fit(gen,epochs=50,callbacks=cbks)
#prior_model.save_model('pretrained_models/vqvae-librispeech/prior.dnn')
#embed()

train_data = joblib.load('ls-codes')
random_start = np.zeros((1,8,8,513))
random_start[0,1,:] = tf.one_hot(train_data[676][0,:] + 1,depth=513,axis=-1).numpy() #Elijo como primer sample uno existente

prior_model = dienen.Model('models/cnn_vqvae_speech_prior.yaml')
prior_model.build()
prior_model.core_model.model.load_weights('../ckpts/prior/weights.50-3.50.hdf5')

n_frames_predict = 256
generated_code_dist = np.zeros((1,n_frames_predict,8,513))
generated_code_dist[0,1,:] = random_start[0,1,:]

for i in range(2,n_frames_predict-1):
    predicted = prior_model.core_model.model.predict(random_start)
    if i<7:
        generated_code_dist[0,i+1,:,:] = predicted[0,i,:,:]
        random_start[0,i+1,:,:] = predicted[0,i,:,:]
    else:
        generated_code_dist[0,i+1,:,:] = predicted[0,-1,:,:]
        random_start[0,1:,:,:] = generated_code_dist[:,i-7:i,:,:]

categorical_distribution = tfp.distributions.Categorical(logits=generated_code_dist)
generated_code = categorical_distribution.sample().numpy()
generated_code -= 1
generated_code[generated_code==-1]=0

decoder_model = dienen.Model('models/cnn_vqvae_speech_decode.yaml')
decoder_model.build()

generated_code_framed = np.concatenate([generated_code[:,i:i+8,:] for i in range(0,n_frames_predict,8)],axis=0)
generated_spectrogram = tf.cast(tf.squeeze(decoder_model.core_model.model.predict(generated_code_framed),axis=-1),tf.complex128)
generated_spectrogram = np.pad(generated_spectrogram,((0,0),(0,0),(0,1)))
#generated_spectrogram = generated_spectrogram*np.exp(1.0j*np.random.uniform(-np.pi,np.pi,size=generated_spectrogram.shape))
#generated_audio = tf.signal.inverse_stft(tf.cast(tf.squeeze(generated_spectrogram,axis=-1),tf.complex128),frame_length=256,frame_step=128,
#                                        window_fn=tf.signal.inverse_stft_window_fn(128,forward_window_fn=tf.signal.hann_window))
generated_spectrogram = np.expand_dims(np.concatenate(generated_spectrogram,axis=0),axis=0)
mag_predictions = tf.abs(generated_spectrogram)
waveforms = tf.signal.inverse_stft(generated_spectrogram,frame_step=128,frame_length=256,
                                    window_fn=tf.signal.inverse_stft_window_fn(128,forward_window_fn=tf.signal.hann_window))
#gl_iters=100
#for i in range(gl_iters):
#    waveforms = tf.signal.inverse_stft(generated_spectrogram,frame_step=128,frame_length=256,
#                                    window_fn=tf.signal.inverse_stft_window_fn(128,forward_window_fn=tf.signal.hann_window))
#    waveforms_stft = tf.signal.stft(waveforms,frame_step=128,frame_length=256)
#    new_phase = tf.math.angle(waveforms_stft)
#    generated_spectrogram = tf.cast(mag_predictions,tf.complex64)*tf.math.exp(tf.math.multiply(tf.complex(0.0,1.0),tf.cast(new_phase,tf.complex64)))

#import librosa
#waveforms = librosa.istft(generated_spectrogram[0].T,hop_length=128,win_length=256,window=tf.signal.hann_window(256).numpy())
import matplotlib.pyplot as plt
import soundfile as sf

plt.imshow(np.abs(generated_spectrogram[0,:,:]).T,aspect='auto',origin='lower')
plt.savefig('generated.png')

sf.write('generated_speech.wav',np.squeeze(waveforms),16000)
