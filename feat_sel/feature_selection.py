import time
t000 = time.time()

import pandas as pd
import xgboost as xgb

# df: DataFrame containing X, y, timestamp
def feature_importance(df, feature_names, target_name, params_xgb, n_round=1,
                       t_begin=0, t_end=905, t_delta=100,
                       t_win_train=100, t_win_valid=100, total_gain=False,
                       seed_offset=123):
    t0 = time.time()
    imp_train = pd.DataFrame(columns=feature_names)
    imp_valid = pd.DataFrame(columns=feature_names)
    for t_left in range(t_begin, t_end-t_win_train-t_win_valid, t_delta):
        t_mid = t_left + t_win_train
        t_right = t_mid + t_win_valid
        seed = t_mid + seed_offset
        params_xgb['seed'] = seed
        train_idx = (df.timestamp >= t_left) & (df.timestamp < t_mid)
        valid_idx = (df.timestamp >= t_mid) & (df.timestamp < t_right)
        X_train = df[feature_names][train_idx].values
        y_train = df[target_name][train_idx].values
        X_valid =  df[feature_names][valid_idx].values
        y_valid = df[target_name][valid_idx].values

        # train
        xgmat_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        bst = xgb.train(params_xgb,
                        xgmat_train,
                        num_boost_round=n_round,
                        verbose_eval=False)

        # refresh
        params_xgb_refresh = params_xgb.copy()
        params_xgb_refresh.update({
            'tree_method' : 'exact', # tree_method=hist produces error
            'process_type': 'update',
            'updater'     : 'refresh',
            'refresh_leaf': False})
        xgmat_valid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
        bst_r = xgb.train(params_xgb_refresh,
                          xgmat_valid,
                          xgb_model=bst,
                          num_boost_round=n_round,
                          verbose_eval=False)

        imp_t = pd.Series(bst.get_score(importance_type='gain'))
        imp_v = pd.Series(bst_r.get_score(importance_type='gain'))
        if total_gain:
            imp_t *= pd.Series(bst.get_score(importance_type='weight'))
            imp_v *= pd.Series(bst_r.get_score(importance_type='weight'))
        imp_t = imp_t.to_frame().T
        imp_v = imp_v.to_frame().T
        imp_train = imp_train.append(imp_t)
        imp_valid = imp_valid.append(imp_v)

    return imp_train.fillna(0).mean(0), imp_valid.fillna(0).mean(0)

# train: df.timestamp<=t_end
# imp_type: 'valid', 'train'
def feature_selection(df, params, topk=10, n_round=5, t_begin=0, t_end=905, t_delta=10,
                      imp_type='valid', winsize=100, save_imp=False, seed_offset=123):
    imp_tr, imp_va = feature_importance(df,
                                        feature_names,
                                        'y',
                                        params,
                                        n_round=n_round,
                                        t_begin=t_begin,
                                        t_end=t_end,
                                        t_delta=t_delta,
                                        t_win_train=winsize,
                                        t_win_valid=winsize,
                                        seed_offset=seed_offset)

    ncol_max_tr = imp_tr[imp_tr>0].size
    ncol_max_va = imp_va[imp_va>0].size
    print("imp_train>0:%d, imp_valid>0:%d" % (ncol_max_tr, ncol_max_va))
    imp = pd.DataFrame({'train':imp_tr, 'valid':imp_va})
    if save_imp:
        imp.to_csv('feat_imp_seed_%d.csv' % seed_offset)
    ncol_max = ncol_max_va if imp_type=='valid' else ncol_max_tr
    print(imp.sort_values(imp_type).tail(10))
    feature_names_sorted = imp.sort_values(imp_type, ascending=False).index
    if topk > ncol_max:
        topk = ncol_max
    return feature_names_sorted[:topk]

if __name__ == '__main__':
    df = pd.read_hdf('../input/train.h5')
    excl = ['id', 'sample', 'y', 'timestamp']
    col = [c for c in df.columns if c not in excl]
    df['znull'] = df[col].isnull().sum(axis=1)
    feature_names = col + ['znull']

    params_xgb_fs = {'objective'       : "reg:linear",
                     'eta'             : 0.1,
                     'subsample'       : 0.9,
                     'colsample_bytree': 1,
                     'max_depth'       : 6,
                     'min_child_weight': 1000,
                     'base_score'      : 0,
                     'silent'          : True,
    }
    topk = 40
    cols_lst = []
    for i in range(30):
        seed = 10191 + 571 * i
        print("\nseed:%d" % seed)
        cols_lst.append(feature_selection(df, params_xgb_fs,
                                          topk=topk,
                                          n_round=5,
                                          t_begin=0,
                                          t_end=df.timestamp.max(),
                                          t_delta=10,
                                          winsize=100,
                                          save_imp=True,
                                          seed_offset=seed))
        print("%.1fs" % (time.time() - t000))
    pd.DataFrame(cols_lst).to_csv('feature_names_top%d.csv' % topk, header=False, index=False)

    print("\nDone: %.1fs" % (time.time() - t000))
    
        
