# coding=UTF-8
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
from hyperopt import fmin, hp, tpe
import hyperopt
from time import clock
import datetime
import sys
from sklearn import metrics
model_name = 'xgb'
n_jobs = 2
def xgb_train(dtrain, dtest, param, offline=True, verbose=True, num_boost_round=1000):
    if verbose:
        if offline:
            watchlist = [(dtrain, 'train'), (dtest, 'test')]
        else:
            watchlist = [(dtrain, 'train')]
        model = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=watchlist)
        feature_score = model.get_fscore()
        feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
        fs = []
        for key, value in feature_score:
            fs.append("{0},{1}\n".format(key, value))
        if offline:
            feature_score_file = './feature_score/offline_feature_score' + '.csv'
        else:
            feature_score_file = './feature_score/online_feature_score' + '.csv'
        f = open(feature_score_file, 'w')
        f.writelines("feature,score\n")
        f.writelines(fs)
        f.close()
    else:
        model = xgb.train(param, dtrain, num_boost_round=num_boost_round)
    return model

def xgb_predict(model, dtest):
    print ('model_best_ntree_limit : {0}\n'.format(model.best_ntree_limit))
    pred_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    return pred_y

def tune_xgb(dtrain, dtest):
    tune_reuslt_file = "./log/tune_" + model_name
    f_w = open(tune_reuslt_file, 'wb')
    def objective(args):
        params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            #'scale_pos_weight': weight,
            # 'lambda': 1000,
            'nthread': n_jobs,
            'eta': args['learning_rate'],
            # 'gamma': args['gamma'],
            'colsample_bytree': args['colsample_bytree'],
            'max_depth': 6, #args['max_depth'],
            'subsample': args['subsample']
        }
        #if fs verbose = False
        model = xgb_train(dtrain, dtest, params, offline=True, verbose=False, num_boost_round=int(args['n_estimators']))

        #model.save_model('xgb.model')
        model.dump_model('dump_model_txt')

        pred_y = xgb_predict(model, dtest)
        test_y = dtest.get_label()
        rmse = np.sqrt(metrics.mean_squared_error(test_y, pred_y))
        xgb_log.write(str(args))
        xgb_log.write('\n')
        xgb_log.write(str(rmse))
        xgb_log.write('\n')
        return rmse
    #import pdb
    #pdb.set_trace()
    # Searching space
    space = {
        'n_estimators': hp.quniform("n_estimators", 100, 200, 20),
        # 'reg_lambda': hp.loguniform("reg_lambda", np.log(1), np.log(1500)),
        # 'gamma': hp.loguniform("gamma", np.log(0.1), np.log(100)),
        'learning_rate': hp.uniform("learning_rate", 0.05, 0.15),
        #'max_depth': hp.choice("max_depth",[4,5,6,7,8,9,10]),
        'subsample': hp.uniform("subsample", 0.5, 0.9),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 0.9),
    }
    #best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=300)
    best_sln = fmin(objective, space, algo=hyperopt.anneal.suggest, max_evals=300)
    pickle.dump(best_sln,f_w,True)
    best_rmse = objective(best_sln)
    xgb_log.write(str(best_rmse) + '\n')
    f_w.close()

def test(dtrain, dtest,best_n_estimators):
    final_result = "./log/xgb_online_result.csv"
    f_w = open(final_result, 'w')
    model = xgb_train(dtrain, dtest, init_params, offline, verbose=False,num_boost_round=best_n_estimators)
    pred_y = xgb_predict(model, dtest)
    res = pd.read_csv('../data/submission_sample.csv')
    res['综合GPA'] = pred_y.flat[:]
    res.to_csv('../data/submission_1.csv',index=False)
    print(pred_y.flat[:])
    #f_w.write(pred_y)
    f_w.close()

def load_data(path, offline=True):
    data = util.load_data(path)
    train, valid = util.partition(data)
    return train, valid

def get_featured_data():
    train_x = pd.read_csv('../data/noEye/train_x.csv')
    train_y = pd.read_csv('../data/noEye/train_y.csv')
    test_x = pd.read_csv('../data/noEye/test_x.csv')
    return train_x, train_y, test_x

if __name__ == '__main__':
    t_start = clock()
    offline = False
    train_x,train_y,test_x = get_featured_data()
    valid_x = train_x[int(0.8*len(train_x)):len(train_x)]
    valid_y = train_y[int(0.8*len(train_y)):len(train_y)]
    #weight = float(len(train_y[train_y==0]))/len(train_y[train_y==1])
    #class_weight = {1:weight,0:1}

    print ('Feature Dims : ')
    print (train_x.shape)
    print (train_y.shape)
    print (test_x.shape)

    dtrainV = xgb.DMatrix(train_x[0:int(0.8*len(train_x))],label=train_y[0:int(0.8*len(train_y))])
    dtrainA = xgb.DMatrix(train_x,label=train_y)
    dtest = xgb.DMatrix(test_x)
    dvalid = xgb.DMatrix(valid_x,label=valid_y)
    del train_x,train_y,test_x
    #gc.collect()

    Auto = True
    if Auto:
        if offline:
            xgb_log = open('./log/xgb_log.txt',mode='w')
            tune_xgb(dtrainV, dvalid)
            xgb_log.close()
        else:
            tune_reuslt_file = "./log/tune_" + model_name
            f_w = open(tune_reuslt_file, 'rb')
            tune_xgb = pickle.load(f_w)
            f_w.close()
            
            best_n_estimators = int(tune_xgb['n_estimators'])
            best_learning_rate = tune_xgb['learning_rate']
            #best_max_depth = int(tune_xgb['max_depth'])
            best_subsample = tune_xgb['subsample']
            best_colsample_bytree = tune_xgb['colsample_bytree']

            init_params = {
                'booster': 'gbtree',
                'objective': 'reg:linear',
                #'scale_pos_weight': weight,
                'max_depth': 6,#best_max_depth,
                'subsample': best_subsample,
                'nthread': n_jobs,
                'eval_metric': 'rmse',
                'colsample_bytree': best_colsample_bytree,
                'eta': best_learning_rate
            }
            test(dtrainA,dtest,best_n_estimators)
    else:
        init_params = {
                'booster': 'gbtree',
                'objective': 'reg:linear',
                'max_depth': 8,
                'eval_metric': 'rmse',
            }
        test(dtrain,dtest,best_n_estimators=300)
        

    t_finish = clock()
    print('==============Costs time : %s s==============' % str(t_finish - t_start))
