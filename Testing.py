from tensorflow.keras.models import model_from_json
import librosa
import numpy as np
import os
path =os.listdir("dataset/COPD/")
# print(path)


json_file = open('GRUmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("GRUmodel.h5")
print("Loaded model from disk")


# for i in path:
#     path1="dataset/COPD/"+i
#     data_x, sampling_rate = librosa.load(path1,res_type='kaiser_fast')
#     mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0)
#     # datalist.append(mfccs)
#     # test=np.array(datalist)
#     # print(test.shape)
#     feat=np.array(mfccs)
    
#     feat=np.reshape(feat,(1,40,1))


#     ypred=model.predict(feat)
#     result=np.argmax(ypred)
#     print(ypred)


datalist=[]
path="dataset/COPD/213_1p5_Pr_mc_AKGC417L.wav"
data_x, sampling_rate = librosa.load(path,res_type='kaiser_fast')
mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0)
# datalist.append(mfccs)
# test=np.array(datalist)
# print(test.shape)
feat=np.array(mfccs)
print(feat.shape)
feat=np.reshape(feat,(1,40,1))


ypred=model.predict(feat)
result=np.argmax(ypred)
print(ypred[0][0])