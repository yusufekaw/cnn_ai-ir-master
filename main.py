import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from Model import Model_CNN_IR
from newDataset.Dataset import load_storage
import tensorflow as tf

if __name__ == '__main__':
    #objek dataset
    load = load_storage()
    print("shape x_train", load.x_train.shape)
    print("shape y_train", load.y_train.shape)

    #model ML
    ML = Model_CNN_IR(load.x_train, load.y_train, load.x_test, load.y_test, 0.001, 2)
    ML.create_architecture()
    ML.train_model()
    ML.model_summary()
    ML.plot_Training()

    #analisa model
    #akurasi terhadap data training
    pred_train = ML.ModelPredict(load.x_train)
    print(load.y_train)
    print(pred_train)
    accuracy_train = accuracy_score(load.y_train, pred_train)
    precision_train = precision_score(load.y_train, pred_train, average='micro')
    recall_train = recall_score(load.y_train, pred_train, average='micro')
    f1_train = f1_score(load.y_train, pred_train, average='micro')
    print("---------------------------------------------")
    

    print("akurasi terhadap data training = ", accuracy_train)
    print("presisi terhadap data training = ", precision_train)
    print("recal terhadap data training = ", recall_train)
    print("f1 score terhadap data training = ", f1_train)

    #akurasi terhadap data test
    pred_test = ML.ModelPredict(load.x_test)
    print(load.y_test)
    print(pred_test)
    accuracy_test = accuracy_score(load.y_test, pred_test)
    precision_test = precision_score(load.y_test, pred_test, average='micro')
    recall_test = recall_score(load.y_test, pred_test, average='micro')
    f1_test = f1_score(load.y_test, pred_test, average='micro')
    print("akurasi terhadap data training = ", accuracy_test)
    print("presisi terhadap data training = ", precision_test)
    print("recal terhadap data training = ", recall_test)
    print("f1 score terhadap data training = ", f1_train)
    print("---------------------------------------------")