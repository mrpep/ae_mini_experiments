import tensorflow as tf

class SpectralLoss(tf.keras.losses.Loss):
    def __init__(self,winsizes=[64,128,256,512,1024],hopsizes=[16,32,64,128,256],weights=[0.2,0.2,0.2,0.2,0.2],log=False,clip_db=False,name='StftLoss'):
        super().__init__(name=name)
        self.winsizes = winsizes
        self.hopsizes = hopsizes
        self.weights = weights
        self.eps=1e-16
        self.clip_db = clip_db
        self.log = log

    def call(self, y_true, y_pred):
        loss = 0.0
        for ws,hs, w in zip(self.winsizes,self.hopsizes,self.weights):
            y_true = tf.cast(y_true,tf.float32)
            y_pred = tf.cast(y_pred,tf.float32)
            y_true_stft=tf.signal.stft(y_true,ws,hs)
            y_pred_stft=tf.signal.stft(y_pred,ws,hs)
            y_true_mag=tf.abs(y_true_stft)
            y_pred_mag=tf.abs(y_pred_stft)
            mag_diff = y_true_mag - y_pred_mag
            #if self.log:
            #    y_pred_mag=tf.math.log(y_pred_mag+self.eps)
            #    y_true_mag=tf.math.log(y_true_mag+self.eps)
            #if self.clip_db:
            #    y_pred_mag = tf.math.maximum(y_pred_mag,self.clip_db)
            #    y_true_mag = tf.math.maximum(y_true_mag,self.clip_db)
            #    y_pred_mag=-(y_pred_mag-self.clip_db)/self.clip_db
            #    y_true_mag=-(y_true_mag-self.clip_db)/self.clip_db
            #    mask_true = tf.cast(tf.math.not_equal(y_true_mag, 0), tf.float32)
            #    masked_squared_error = (mask_true * (y_true_mag - y_pred_mag))**2
            #    masked_mse = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(mask_true)
            #    loss = loss + w*masked_mse
            #else:
            loss = loss + w*tf.math.reduce_mean(mag_diff**2)
        return loss