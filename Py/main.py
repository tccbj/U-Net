#  coding:utf8
from model import *
from data import *
from project_img import *
from keras import backend as K
import keras
import pickle
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LossHistory(keras.callbacks.Callback):
    #或者参考https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
    #https://stackoverflow.com/questions/49901026/how-to-check-the-learning-rate-with-train-on-batch-keras
    def on_epoch_end(self, epoch, logs=None):
        print('lr:', K.eval(self.model.optimizer.lr))
        
def myTrain(unet, outweights, train_image_path, train_mask_path, val_image_path, val_mask_path, 
                                                train_batch_size=4, 
                                                val_batch_size=4,
                                                pretrained_weights = None, 
                                                log_path=None, 
                                                epochs=10, 
                                                train_step=295,
                                                val_step=50):
    
    #------训练集生成器------
    myGene = dataGenerator(train_image_path, train_mask_path, batch_size=train_batch_size)
    #------验证集生成器------
    valGene = dataGenerator(val_image_path, val_mask_path, batch_size=val_batch_size)
    #------网络模型------
    model = unet(pretrained_weights = pretrained_weights)
    #------日志记录------
    csv_logger = CSVLogger(log_path)
    #------预加载已训练模型------
 #  model=load_model(r'F:\unet-cbj\unet3.hdf5', custom_objects={'jaccard_coef':jaccard_coef, 
 #                                                                'dice_coef':dice_coef,
 #                                                                'be_dice_loss':be_dice_loss})
    #------编译模型------
    model.compile(optimizer = Adam(lr = 1e-4), loss = be_dice_loss, metrics = ['accuracy',my_accuracy,jaccard_coef, dice_coef])
    #------保存点------
    #outweights = r'F:\unet-cbj\record\unet_pal-{epoch:02d}-{val_dice_coef:.4f}.hdf5'
    #model_checkpoint = ModelCheckpoint(outweights, monitor='loss',verbose=1, save_best_only=False, mode='min')
    model_checkpoint = ModelCheckpoint(outweights, monitor='val_my_accuracy',verbose=1, save_best_only=True, mode='max')
    #------训练要求------
    #early_stop = EarlyStopping(monitor='val_loss', patience=6, mode='min' )
    #lrate = LearningRateScheduler(lr_decay)
    lr_auto = ReduceLROnPlateau(monitor='val_my_accuracy', patience=6, verbose=0,mode='max',factor=0.1, min_lr=1e-9)
    show_lr = LossHistory()
    #------训练时记录的信息与要求------
    call_backs = [csv_logger, show_lr, model_checkpoint,lr_auto]#,early_stop]
    #------训练模型------
    history = model.fit_generator(myGene, steps_per_epoch=train_step, epochs=epochs, callbacks=call_backs, validation_data=valGene, validation_steps=val_step)#, initial_epoch=49)
    return outweights,history

def myPredict(unet, weights, predict_path, batch_size=8, steps=25):
    #------预测生成器------
    predictGene = predictGenerator(predict_path, batch_size=batch_size)
    #------网络模型------
    model = unet(pretrained_weights = weights)
    #------预测训练模型------
    results = model.predict_generator(predictGene,steps=steps,verbose=1)
    return results
    
def dice_coef(y_true, y_pred):
    smooth = 1e-12
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def be_dice_loss(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)-K.log(dice_coef(y_true, y_pred))

def jaccard_coef(y_true, y_pred):
    #__source__:https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
        smooth = 1e-12
        intersection = K.sum(y_true * y_pred, axis=[-1, -2, -3]) 
        sum_ = K.sum(y_true + y_pred, axis=[-1, -2, -3])
        jac = (intersection + smooth)/(sum_ - intersection +smooth)
        return K.mean(jac)
    
def my_accuracy(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    acc = K.abs(y_true_f-y_pred_f)
    return 1-K.mean(acc)

def runall(unet, train_image_path, train_mask_path, val_image_path, val_mask_path, eva_image_path, eva_mask_path, outweights, log_path, save_path, pkl_path, epochs, pretrained_weights=None):
    weights, his = myTrain(unet, outweights, train_image_path, train_mask_path, val_image_path, val_mask_path, pretrained_weights = pretrained_weights, log_path=log_path, epochs=epochs)
#    result = myEvaluate(unet, weights, val_image_path, val_mask_path)
#    #print('evaluate:',result)
#    predict_path = val_image_path
#    predict = myPredict(unet, weights, predict_path)
#    f=open(pkl_path,'wb')
#    pickle.dump(predict,f,0)
#    f.flush()
#    f.close()
#    r = adjustResult(predict)
#    #label = 
#    #best_p,best_acc = find_p(predict,label)
#    #print('best_p:',best_p)
#    #print('best_acc:',best_acc)
#    in_path = val_mask_path
#    saveResultPrj(in_path,save_path,r)
    return weights#result
    
if __name__ == '__main__':
    train_image_path = r'E:\tree\randomdata2\train\imagexx'
    train_mask_path = r'E:\tree\randomdata2\train\label'
    val_image_path = r'E:\tree\randomdata2\val\imagexx'
    val_mask_path = r'E:\tree\randomdata2\val\label'
    eva_image_path = r'E:\tree\randomdata2\test\imagexx'
    eva_mask_path = r'E:\tree\randomdata2\test\label'
    outweights = r'E:\tree\result\record_20190125\unet_xx.hdf5'
    log_path = r'E:\tree\result\record_20190125\train_xx.log'
    save_path = r'E:\tree\result\record_20190125\pxx'
    pkl_path = r'E:\tree\result\record_20190125\predictxx.pkl'
    pretrained_weights = None#r'E:\tree\result\record_20190122\unet_1.hdf5'
    epochs = 100
    unet = unet2
#    result = runall(unet, train_image_path, train_mask_path, val_image_path, val_mask_path, eva_image_path, eva_mask_path, outweights, log_path, save_path, pkl_path, epochs, pretrained_weights)
    result = []
    for i in range(1,9):
        tmp_train_image_path = train_image_path.replace('xx',str(i))
        tmp_val_image_path = val_image_path.replace('xx',str(i))
        tmp_eva_image_path = eva_image_path.replace('xx',str(i))
        tmp_outweights = outweights.replace('xx',str(i))
        tmp_log_path = log_path.replace('xx',str(i))
        tmp_save_path = save_path.replace('xx',str(i))
        tmp_pkl_path = pkl_path.replace('xx',str(i))
        result.append(runall(unet, tmp_train_image_path, train_mask_path, tmp_val_image_path, val_mask_path, tmp_eva_image_path, eva_mask_path, tmp_outweights, tmp_log_path, tmp_save_path, tmp_pkl_path, epochs,pretrained_weights))
    
    print('unet:','unet\n','result:',result,'\nepochs:',epochs)
