from IPython import embed
from dienen.core.file import load_model
import dienen
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import librispeech
from tqdm import tqdm
from dsp import *

#Turn weights and config into dienen model:

#config_path = 'models/cnn_vqvae_speech_stftin.yaml'
#weights_path = '../ckpts/weights.02-0.06.hdf5'
#vqvae_model = dienen.Model(config_path)
#vqvae_model.build()
#vqvae_model.core_model.model.load_weights(weights_path)
#vqvae_model.save_model('pretrained_models/cnn_vqvae_speech_stftin/model.dnn')

#loaded_model = load_model('pretrained_models/cnn_vqvae_speech_stftin/model.dnn')
#loaded_model.core_model.model.summary()
#embed()

#Make training data for prior model:
#vqvae_encoder = dienen.Model('models/cnn_vqvae_speech_stftin_encode.yaml')
#vqvae_encoder.build()
#embed()

#import librispeech
#audio_train_data, audio_test_data = librispeech.read_librispeech('../LibriSpeech', winsize=8385, calculate_spectrogram=True, frame_size=256,hop_size=64)
#batch_size = 64
#audio_train_gen = librispeech.LSGenerator('../LibriSpeech',batch_size=batch_size,winsize=8385)
#audio_train_indexs = []
#for i in tqdm(range(len(audio_train_gen))):
#    audio_train_indexs.append(vqvae_encoder.core_model.model.predict(audio_train_gen.__getitem__(i)[0]))
#audio_train_indexs = vqvae_encoder.core_model.model.predict(audio_train_gen)
#joblib.dump(audio_train_indexs,'ls-codes-pghi')
#Train prior:
#train_data = joblib.load('ls-codes-pghi')
#train_data = np.concatenate(train_data,axis=0)

#class train_generator(tf.keras.utils.Sequence):
#    def __init__(self,data,batch_size,type='spectrogram'):
#        self.data = data
#        self.index = np.arange(0,len(data))
#        self.batch_size = batch_size
#        self.type = type

#    def __getitem__(self,i):
#        batch_idxs = np.take(self.index,np.arange(i*self.batch_size,(i+1)*self.batch_size))
#        batch_data = self.data[batch_idxs] + 1
#        oh_data = tf.one_hot(batch_data,depth=513,axis=-1)
#        x = tf.roll(oh_data,1,axis=1)
#        x = x.numpy()
#        if self.type == 'spectrogram':
#            x[:,0,:,:] = 0
#        else:
#            x[:,0,:] = 0
       
#        return x, oh_data.numpy()
        
#    def on_epoch_end(self):
#        self.index = np.random.permutation(self.index)

#    def __len__(self):
#        return len(self.data)//self.batch_size

#gen = train_generator(train_data,batch_size = 64,type='raw')

#prior_model = dienen.Model('models/cnn_vqvae_speech_stftin_prior.yaml')
#prior_model.build()
#prior_model.core_model.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='categorical_accuracy')
#cbks = [tf.keras.callbacks.ModelCheckpoint('../ckpts/prior-pghi/weights.{epoch:02d}-{loss:.2f}.hdf5')]
#prior_model.core_model.model.fit(gen,epochs=50,callbacks=cbks)
#prior_model.save_model('pretrained_models/cnn_vqvae_speech_stftin/prior.dnn')
#embed()
#data_type = 'raw'

data_type='spectrogram'
do_pghi = True

train_data = joblib.load('ls-codes-pghi')
train_data = np.concatenate(train_data,axis=0)

#if data_type == 'spectrogram':
#    random_start = np.zeros((1,8,8,513))
#    random_start[0,1,:] = tf.one_hot(train_data[877][0,:],depth=513,axis=-1).numpy() #Elijo como primer sample uno existente
#else:
#    random_start = np.zeros((1,31,513))
#    random_start[0,1] = tf.one_hot(train_data[877][0],depth=513,axis=-1).numpy() #Elijo como primer sample uno existente

n_frames_predict = 248
steps_per_prediction = 8

model_input = np.zeros((n_frames_predict//steps_per_prediction + 2,8,8,513))

prior_model = dienen.Model('models/cnn_vqvae_speech_stftin_prior.yaml')
prior_model.build()
prior_model.modify([{'ar_lstm/stateful':True},{'delete': 'input_indexs/shape'},{'input_indexs/batch_shape':model_input.shape}])
prior_model.core_model.model.load_weights('../ckpts/prior-pghi/weights.19-2.80.hdf5')

model_input[0,:,:] = tf.one_hot(train_data[877]+1,depth=513,axis=-1).numpy()
generated_code = np.zeros((1,n_frames_predict,8,513))
for i in range(0,n_frames_predict):
    batch_idx = i//steps_per_prediction + 1
    step_idx = i%steps_per_prediction
    predicted = prior_model.core_model.model.predict(model_input,batch_size=len(model_input))
    sampled_idxs = tfp.distributions.Categorical(logits=predicted[batch_idx,step_idx,:,:]).sample().numpy()
    code_i = tf.one_hot(sampled_idxs,depth=513,axis=-1).numpy()
    generated_code[0,i,:,:] = code_i
    if step_idx == steps_per_prediction-1:
        step_idx = 0
        batch_idx += 1
    model_input[batch_idx,step_idx,:]=code_i

generated_code = np.argmax(generated_code,axis=-1) - 1
#if data_type == 'spectrogram':
#    generated_code_dist = np.zeros((1,n_frames_predict,8,513))
#    generated_code_dist[0,1,:] = random_start[0,1,:]
#else:
#    generated_code_dist = np.zeros((1,n_frames_predict,513))
#    generated_code_dist[0,1,:] = random_start[0,1,:]


#for i in range(0,n_frames_predict-1):
#    predicted = prior_model.core_model.model.predict(random_start)
#    if i<steps_per_prediction-1:
#        if data_type == 'spectrogram':
#            generated_code_dist[0,i+1,:,:] = predicted[0,i,:,:]
#            random_start[0,i+1,:,:] = predicted[0,i,:,:]
#        else:
#            generated_code_dist[0,i+1,:] = predicted[0,i,:]
#            random_start[0,i+1,:] = predicted[0,i,:]
#    else:
#        if data_type == 'spectrogram':
#            generated_code_dist[0,i+1,:,:] = predicted[0,-1,:,:]
#            random_start[0,:,:,:] = generated_code_dist[:,i-steps_per_prediction:i,:,:]
#        else:
#            generated_code_dist[0,i+1,:] = predicted[0,-1,:]
#            random_start[0,:,:] = generated_code_dist[:,i-steps_per_prediction:i,:]

#categorical_distribution = tfp.distributions.Categorical(logits=generated_code_dist)
#generated_code = categorical_distribution.sample().numpy()
#generated_code -= 1
#generated_code[generated_code==-1]=0

decoder_model = dienen.Model('models/cnn_vqvae_speech_stftin_decode.yaml')
decoder_model.build()

import matplotlib.pyplot as plt
import soundfile as sf

if data_type == 'spectrogram':
    generated_code_framed = np.concatenate([generated_code[:,i:i+steps_per_prediction,:] for i in range(0,n_frames_predict-steps_per_prediction,steps_per_prediction//2)],axis=0)
    generated_spectrogram = tf.squeeze(decoder_model.core_model.model.predict(generated_code_framed),axis=-1)
    generated_spectrogram = np.pad(generated_spectrogram,((0,0),(0,0),(0,1)))
    generated_spectrogram_ = np.zeros((generated_spectrogram.shape[0]*generated_spectrogram.shape[1],generated_spectrogram.shape[2]))
    for i,frame in enumerate(generated_spectrogram):
        try:
            generated_spectrogram_[i*64:i*64 + 128] += np.expand_dims(np.hanning(128),axis=-1)*frame
        except:
            embed()
    #generated_spectrogram = np.expand_dims(np.concatenate(generated_spectrogram,axis=0),axis=0)
    mag_predictions = np.squeeze(tf.abs(generated_spectrogram_))

    if do_pghi:
        estimated_phase = pghi(mag_predictions,256,64)
        waveforms = istft(mag_predictions*np.exp(1.0j*estimated_phase),256,64)
    else:
        waveforms = tf.signal.inverse_stft(generated_spectrogram_,frame_step=128,frame_length=256,
                                        window_fn=tf.signal.inverse_stft_window_fn(128,forward_window_fn=tf.signal.hann_window))

    plt.imshow(np.abs(generated_spectrogram_[:,:]).T,aspect='auto',origin='lower')
    plt.savefig('generated.png')
#else:
#    generated_code_framed = np.concatenate([generated_code[:,i:i+steps_per_prediction] for i in range(0,n_frames_predict,steps_per_prediction)],axis=0)
#    silence_artifact = np.zeros_like(generated_code_framed)
#    silence_artifact[:,:] = 222
#    waveforms = decoder_model.core_model.model.predict(generated_code_framed) - decoder_model.core_model.model.predict(silence_artifact)
#    waveforms = np.concatenate(waveforms,axis=0)
#    waveforms = tf.signal.overlap_and_add(waveforms,frame_step=128)
    
sf.write('generated_speech.wav',np.squeeze(waveforms),16000)
#generated_spectrogram = generated_spectrogram*np.exp(1.0j*np.random.uniform(-np.pi,np.pi,size=generated_spectrogram.shape))
#generated_audio = tf.signal.inverse_stft(tf.cast(tf.squeeze(generated_spectrogram,axis=-1),tf.complex128),frame_length=256,frame_step=128,
#                                        window_fn=tf.signal.inverse_stft_window_fn(128,forward_window_fn=tf.signal.hann_window))

#gl_iters=100
#for i in range(gl_iters):
#    waveforms = tf.signal.inverse_stft(generated_spectrogram,frame_step=128,frame_length=256,
#                                    window_fn=tf.signal.inverse_stft_window_fn(128,forward_window_fn=tf.signal.hann_window))
#    waveforms_stft = tf.signal.stft(waveforms,frame_step=128,frame_length=256)
#    new_phase = tf.math.angle(waveforms_stft)
#    generated_spectrogram = tf.cast(mag_predictions,tf.complex64)*tf.math.exp(tf.math.multiply(tf.complex(0.0,1.0),tf.cast(new_phase,tf.complex64)))

#import librosa
#waveforms = librosa.istft(generated_spectrogram[0].T,hop_length=128,win_length=256,window=tf.signal.hann_window(256).numpy())




