import joblib
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
class Main(object):
    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()
        self.graph = tf.compat.v1.get_default_graph()
        set_session(self.sess)
        self.model = load_model('ann-churn-model.h5')
    def predict(self, X):
        with self.graph.as_default():
            set_session(self.sess)
            X = json.loads(X)
            X_new = []
            for value in X.values():
                X_new.append(value)
            X_new=np.array([X_new])
            label = joblib.load('label_encoder.joblib')
            X_new[:, 2] = label.transform(X_new[:, 2])  #label encode gender column
            saved_encoder = joblib.load('onehot_encoder.joblib')
            encoded_values = saved_encoder.transform(X_new[:, [1]]).toarray()
            X_new = np.delete(X_new, 1, axis=1)     #remove nonencoded country column
            X_new = np.hstack((encoded_values, X_new))      #concatenate because OneHotEncoder pushes encoded values to beginning of array
            sc = joblib.load('scaler.bin')
            result = self.model.predict(sc.transform(X_new))
            return json.dumps(result.tolist())