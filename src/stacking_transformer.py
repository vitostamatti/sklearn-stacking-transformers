import numpy as np
import scipy.stats as st
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import has_fit_parameter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder


class _BaseStackingTransformer():

    def __init__(self,
                 estimators=None,
                 metric=None,
                 n_folds=4,
                 shuffle=False,
                 stratified=False,
                 random_state=0,
                 n_jobs = 1,
                 verbose=0):

        self.estimators = estimators
        self.metric = metric
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.stratified =stratified
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.estimators_ = [(name, clone(estim)) 
            for name, estim in self.estimators]

        if self.verbose not in [0, 1]:
            raise ValueError('Parameter ``verbose`` must be 0, 1')


    def fit(self, X, y):
        """Fit all base estimators.

        Args:
            X (np.array): 2d numpy array or sparse matrix of shape [n_samples, n_features]
            y (np.array): 1d numpy array of shape [n_samples].Target values

        Returns:
            RegressionStackingTransformer: fitted transformer 
        """  
        return self._fit(X, y)


    def transform(self, X):
        """
        Transform (predict) given data set.
        If X is train set: for each estimator return out-of-fold predictions (OOF).
        If X is any other set: for each estimator return mean (mode) of predictions made in each fold
      
        Args:
            X (np.array): 2d numpy array or sparse matrix of shape [n_samples, n_features]

        Returns:
            np.array: 2d numpy array of shape [n_samples, n_estimators]
        """
        return self._transform(X)


    def fit_transform(self, X, y):
        """Fit all base estimators and transform (predict) train set.

        Args:
            X (np.array): 2d numpy array or sparse matrix of shape [n_samples, n_features]
            y (np.array): 1d numpy array of shape [n_samples].Target values

        Returns:
            np.array: 2d numpy array of shape [n_samples, n_estimators]
        """  
        return self.fit(X, y).transform(X)


    def _random_choice(self, n, size, bound=2**30):
        """
        Random select size elements from a list 
        of n ints without replacement
        """
        ids = [i for i in range(n)]        
        return np.random.choice(ids, size=size, replace=False)


    def _get_footprint(self, X, n_items=1000):
        """Selects n_items random elements 
        from 2d numpy array or matrix.
        """
        footprint = []
        r, c = X.shape
        n = r * c
        ids = self._random_choice(n, min(n_items, n))

        for i in ids:
            row = i // c 
            col = i - row * c
            footprint.append((row, col, X[row, col]))

        return footprint

     
    def _check_identity(self, X,rtol=1e-05, atol=1e-08, equal_nan=False):
        """Checks 2d numpy array or sparse matrix identity
        by its shape and footprint.
        """
        if X.shape != self.train_shape_:
            return False

        return all([
                np.isclose(
                X[coo[0], coo[1]], 
                coo[2], rtol=rtol, 
                atol=atol, equal_nan=equal_nan
                ) for coo in self.train_footprint_
            ])


    def _get_params(self, attr, deep=True):
        """Gives ability to get parameters of nested estimators
        """
        out = super(RegressionStackingTransformer, self).get_params(deep=False)
        if not deep:
            return out
        estimators = getattr(self, attr)
        if estimators is None:
            return out
        out.update(estimators)
        for name, estimator in estimators:
            for key, value in iter(estimator.get_params(deep=True).items()):
                out['%s__%s' % (name, key)] = value
        return out


    def get_params(self, deep=True):
        return self._get_params('estimators', deep=deep)


    def is_train_set(self, X):
        """Checks if given data set was used to train.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csr'], force_all_finite=True)
        return self._check_identity(X)


    def _unpack_parallel_preds(self, yps):
        y_m = [np.concatenate(yps[self.n_folds*i:self.n_folds*(i+1)], axis=0) 
            for i in range(self.n_estimators_)]
        y_out = np.concatenate([y[:,None] for y in y_m], axis=1)
        return y_out    


    def _unpack_parallel_scores(self,scores):
        scores_out = np.array([
            np.array(scores[self.n_folds*i:self.n_folds*i+self.n_folds]) 
            for i in range(self.n_estimators_)]
            )
        return scores_out           


    def _compute_estimators_scores(self):
        estimators_scores = []
        for estimator_counter in range(self.n_estimators_):
            estim_name = self.estimators_[estimator_counter][0]
            estim_mean = np.mean(self.scores_[estimator_counter])
            estim_std = np.std(self.scores_[estimator_counter])
            estimators_scores.append((estim_name, estim_mean, estim_std))

            if self.verbose > 0:
                estim_name = 'Estimator: [%s: %s]' % (self.estimators_[estimator_counter][0], self.estimators_[estimator_counter][1].__class__.__name__)
                mean_str = 'Mean Scores: [%.8f]  -  Std Scrores: [%.8f]\n' % (estim_mean, estim_std)
                print(estim_name)
                print(mean_str)

        return estimators_scores 


    def _fit_predict_models(self, X, y, models, metric, kf):

        def _fit_predict_single_model(m, xtr, ytr, xte, yte, metric):
            m.fit(xtr, ytr)
            yp = m.predict(xte)
            score = metric(yte, yp)
            return m, score            

        results = Parallel(n_jobs=self.n_jobs)(delayed(_fit_predict_single_model)(
                models[estimator_counter][fold_counter], 
                X[tr_index], 
                y[tr_index], 
                X[te_index], 
                y[te_index], 
                metric
            ) 
            for estimator_counter in range(len(self.estimators_))
            for fold_counter, (tr_index, te_index) in enumerate(kf.split(X, y)) 
            )

        return results


    def _predict_models(self, X, models, kf, training=True):

        def _predict_single_model(m, xte):
            yp = m.predict(xte)
            return yp     

        if training:
            results = Parallel(n_jobs=self.n_jobs)(delayed(_predict_single_model)(
                    models[estimator_counter][fold_counter], 
                    X[te_index]
                ) 
                for estimator_counter in range(len(self.estimators_))
                for fold_counter, (tr_index, te_index) in enumerate(kf.split(X, self._y_)) 
                )
        else:
            results = Parallel(n_jobs=self.n_jobs)(delayed(_predict_single_model)(
                    models[estimator_counter][fold_counter], 
                    X
                ) 
                for estimator_counter in range(len(self.estimators_))
                for fold_counter in range(self.n_folds)
                )

        return results



class ClassificationStackingTransformer(BaseEstimator, TransformerMixin, _BaseStackingTransformer):

    def _fit(self, X, y):
        X, y = check_X_y(X, y,accept_sparse=['csr'],  
                         force_all_finite=True, multi_output=False) 

        self.train_shape_ = X.shape
        self.n_train_examples_ = X.shape[0]
        self.n_features_ = X.shape[1]

        self.n_estimators_ = len(self.estimators_)
        self.train_footprint_ = self._get_footprint(X)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.n_classes_ = len(self.classes_)
        y = self.le_.transform(y)

        if self.metric is None:
            self.metric_ = accuracy_score

        if self.verbose > 0:
            metric_str = 'metric:  %s' % self.metric_.__name__
            n_estimators_str = 'n_estimators:  %d' % self.n_estimators_
            print(metric_str,"\n",n_estimators_str,'\n')

        if self.shuffle:
            X, y = shuffle(X, y, random_state=self.random_state)

        if self.stratified:
            self.kf_ = StratifiedKFold(n_splits=self.n_folds,shuffle=False)
        else:
            self.kf_ = KFold(n_splits=self.n_folds,shuffle=False)
        self._y_ = None

        self.models_ = []
        for n, est in self.estimators_:
            self.models_.append([clone(est) for _ in range(self.n_folds)])
      
        results = self._fit_predict_models(X, y, self.models_, self.metric_, self.kf_)

        models, scores = [r[0] for r in results], [r[1] for r in results]

        self.models_ = [models[self.n_folds*i:self.n_folds*i+self.n_folds] for i in range(self.n_estimators_)]
            
        self.scores_ = self._unpack_parallel_scores(scores) 
    
        self.mean_std_ = self._compute_estimators_scores()

        return self


    def _transform(self, X):
        """Transform (predict) given data set.
        """
        X = check_array(X, accept_sparse=['csr'], force_all_finite=True)
        is_train_set = self._check_identity(X)

        if self.verbose > 0:
            if is_train_set:
                print('Train set was detected.')

        results = self._predict_models(X, self.models_, self.kf_, training=is_train_set)

        if is_train_set:
            S_train = self._unpack_parallel_preds(results)
            return S_train

        else:
            folds_preds_test = np.array([
                np.array(results[self.n_folds*i:self.n_folds*i+self.n_folds]) 
                for i in range(self.n_estimators_)
                ])

            # S_test = np.mean(folds_preds_test, axis=1).T
            S_test = st.mode(folds_preds_test, axis=1)[0].T
            S_test = S_test[:,0,:] 

            return S_test



class RegressionStackingTransformer(BaseEstimator,TransformerMixin,_BaseStackingTransformer):
    """_summary_

    Args:
        estimators (list, optional): list of tuples. Defaults to None.
            Base level estimators.You can use any sklearn-like estimators.
            Each tuple in the list contains arbitrary unique name and estimator object. 

        metric (callable, optional): Evaluation metric (score function) which is used to calculate. Defaults to None.
            If None then ``mean_squared_error`` is used.

        n_folds (int, optional): number of folds. Defaults to 4.

        shuffle (bool, optional): if shuffle data before fitting estimators. Defaults to False.

        random_state (int, optional): Defaults to 0.

        n_jobs (int, optional): number of jobs tu run in parallel. Defaults to 1.

        verbose (int, optional): Posible values in [0,1]. Defaults to 0.

    """


    def _fit(self, X, y):
        """Fit all base estimators using parallel
        """
        X, y = check_X_y(X, y,
                         accept_sparse=['csr'],  
                         force_all_finite=True, 
                         multi_output=False) 

        self.train_shape_ = X.shape
        self.n_train_examples_ = X.shape[0]
        self.n_features_ = X.shape[1]

        self.n_classes_ = None
        self.n_estimators_ = len(self.estimators_)
        self.train_footprint_ = self._get_footprint(X)

        if self.metric is None:
            self.metric_ = mean_squared_error

        if self.verbose > 0:
            metric_str = 'metric:  %s' % self.metric_.__name__
            n_estimators_str = 'n_estimators:  %d' % self.n_estimators_
            print(metric_str,"\n",n_estimators_str,'\n')

        if self.shuffle:
            X, y = shuffle(X, y, random_state=self.random_state)

        self.kf_ = KFold(n_splits=self.n_folds,shuffle=False)
        self._y_ = None

        self.models_ = []
        for n, est in self.estimators_:
            self.models_.append([clone(est) for _ in range(self.n_folds)])
   
        results = self._fit_predict_models(X, y, self.models_, self.metric_, self.kf_)

        models, scores = [r[0] for r in results], [r[1] for r in results]

        self.models_ = [models[self.n_folds*i:self.n_folds*i+self.n_folds] for i in range(self.n_estimators_)]
            
        self.scores_ = self._unpack_parallel_scores(scores) 
    
        self.mean_std_ = self._compute_estimators_scores()

        return self


    def _transform(self, X):
        """Transform (predict) given data set.
        """
        X = check_array(X, accept_sparse=['csr'], force_all_finite=True)
        is_train_set = self._check_identity(X)

        if self.verbose > 0:
            if is_train_set:
                print('Train set was detected.')

        def _predict_model(m, xte):
            yp = m.predict(xte)
            return yp 

        results = self._predict_models(X, self.models_, self.kf_, training=is_train_set)

        if is_train_set:

            S_train = self._unpack_parallel_preds(results)
            return S_train

        else:

            folds_preds_test = np.array([
                np.array(results[self.n_folds*i:self.n_folds*i+self.n_folds]) 
                for i in range(self.n_estimators_)
                ])

            S_test = np.mean(folds_preds_test, axis=1).T

            return S_test





# class RegressionStackingTransformer(BaseEstimator,TransformerMixin):
#     """_summary_

#     Args:
#         estimators (list, optional): list of tuples. Defaults to None.
#             Base level estimators.You can use any sklearn-like estimators.
#             Each tuple in the list contains arbitrary unique name and estimator object. 

#         metric (callable, optional): Evaluation metric (score function) which is used to calculate. Defaults to None.
#             If None then ``mean_squared_error`` is used.

#         n_folds (int, optional): number of folds. Defaults to 4.

#         shuffle (bool, optional): if shuffle data before fitting estimators. Defaults to False.

#         random_state (int, optional): Defaults to 0.

#         n_jobs (int, optional): number of jobs tu run in parallel. Defaults to 1.

#         verbose (int, optional): Posible values in [0,1]. Defaults to 0.

#     """

#     def __init__(self,
#                  estimators=None,
#                  metric=None,
#                  n_folds=4,
#                  shuffle=False,
#                  random_state=0,
#                  n_jobs = 1,
#                  verbose=0):


#         self.estimators = estimators
#         self.metric = metric
#         self.n_folds = n_folds
#         self.shuffle = shuffle
#         self.random_state = random_state
#         self.verbose = verbose
#         self.n_jobs = n_jobs

#         self.estimators_ = [
#             (name, clone(estim)) 
#             for name, estim in self.estimators
#         ]

#         if self.verbose not in [0, 1]:
#             raise ValueError('Parameter ``verbose`` must be 0, 1')

#     def fit(self, X, y):
#         """Fit all base estimators.

#         Args:
#             X (np.array): 2d numpy array or sparse matrix of shape [n_samples, n_features]
#             y (np.array): 1d numpy array of shape [n_samples].Target values

#         Returns:
#             RegressionStackingTransformer: fitted transformer 
#         """  
#         return self._fit_parallel(X, y)


#     def transform(self, X):
#         """
#         Transform (predict) given data set.
#         If X is train set: for each estimator return out-of-fold predictions (OOF).
#         If X is any other set: for each estimator return mean (mode) of predictions made in each fold
      
#         Args:
#             X (np.array): 2d numpy array or sparse matrix of shape [n_samples, n_features]

#         Returns:
#             np.array: 2d numpy array of shape [n_samples, n_estimators]
#         """
#         return self._transform_parallel(X)
    

#     def fit_transform(self, X, y):
#         """Fit all base estimators and transform (predict) train set.

#         Args:
#             X (np.array): 2d numpy array or sparse matrix of shape [n_samples, n_features]
#             y (np.array): 1d numpy array of shape [n_samples].Target values

#         Returns:
#             np.array: 2d numpy array of shape [n_samples, n_estimators]
#         """  
#         return self.fit(X, y).transform(X)


#     def _unpack_parallel_preds(self, yps):
#         y_m = [np.concatenate(yps[self.n_folds*i:self.n_folds*(i+1)], axis=0) 
#             for i in range(self.n_estimators_)]
#         y_out = np.concatenate([y[:,None] for y in y_m], axis=1)
#         return y_out    


#     def _unpack_parallel_scores(self,scores):
#         scores_out = np.array([
#             np.array(scores[self.n_folds*i:self.n_folds*i+self.n_folds]) 
#             for i in range(self.n_estimators_)]
#             )
#         return scores_out           


#     def _compute_estimators_scores(self):
#         estimators_scores = []
#         for estimator_counter in range(self.n_estimators_):
#             estim_name = self.estimators_[estimator_counter][0]
#             estim_mean = np.mean(self.scores_[estimator_counter])
#             estim_std = np.std(self.scores_[estimator_counter])
#             estimators_scores.append((estim_name, estim_mean, estim_std))

#             if self.verbose > 0:
#                 estim_name = 'Estimator: [%s: %s]' % (self.estimators_[estimator_counter][0], self.estimators_[estimator_counter][1].__class__.__name__)
#                 mean_str = 'Mean Scores: [%.8f]  -  Std Scrores: [%.8f]\n' % (estim_mean, estim_std)
#                 print(estim_name)
#                 print(mean_str)

#         return estimators_scores


#     def _fit_parallel(self, X, y):
#         """Fit all base estimators using parallel
#         """
#         X, y = check_X_y(X, y,
#                          accept_sparse=['csr'],  
#                          force_all_finite=True, 
#                          multi_output=False) 

#         self.train_shape_ = X.shape
#         self.n_train_examples_ = X.shape[0]
#         self.n_features_ = X.shape[1]

#         self.n_classes_ = None
#         self.n_estimators_ = len(self.estimators_)
#         self.train_footprint_ = self._get_footprint(X)

#         if self.metric is None:
#             self.metric_ = mean_squared_error

#         if self.verbose > 0:
#             metric_str = 'metric:  %s' % self.metric_.__name__
#             n_estimators_str = 'n_estimators:  %d' % self.n_estimators_
#             print(metric_str,"\n",n_estimators_str,'\n')

#         if self.shuffle:
#             X, y = shuffle(X, y, random_state=self.random_state)

#         self.kf_ = KFold(n_splits=self.n_folds,shuffle=False)
#         self._y_ = None

#         self.models_ = []
#         for n, est in self.estimators_:
#             self.models_.append([clone(est) for _ in range(self.n_folds)])
   

#         def _fit_predict_model(m, xtr, ytr, xte, yte, metric):
#             m.fit(xtr, ytr)
#             yp = m.predict(xte)
#             score = metric(yte, yp)
#             return m, score            

#         results = Parallel(n_jobs=self.n_jobs)(delayed(_fit_predict_model)(
#                 self.models_[estimator_counter][fold_counter], 
#                 X[tr_index], 
#                 y[tr_index], 
#                 X[te_index], 
#                 y[te_index], 
#                 self.metric_
#             ) 
#             for estimator_counter in range(len(self.estimators_))
#             for fold_counter, (tr_index, te_index) in enumerate(self.kf_.split(X, y)) 
#             )

#         models, scores = [r[0] for r in results], [r[1] for r in results]

#         self.models_ = [models[self.n_folds*i:self.n_folds*i+self.n_folds] for i in range(self.n_estimators_)]
            
#         self.scores_ = self._unpack_parallel_scores(scores) 
    
#         self.mean_std_ = self._compute_estimators_scores()

#         return self


#     def _transform_parallel(self, X):
#         """Transform (predict) given data set.
#         """
#         X = check_array(X, accept_sparse=['csr'], force_all_finite=True)
#         is_train_set = self._check_identity(X)

#         if self.verbose > 0:
#             if is_train_set:
#                 print('Train set was detected.')

#         def _predict_model(m, xte):
#             yp = m.predict(xte)
#             return yp 

#         if is_train_set:
#             results = Parallel(n_jobs=self.n_jobs)(delayed(_predict_model)(
#                     self.models_[estimator_counter][fold_counter], 
#                     X[te_index]
#                 ) 
#                 for estimator_counter in range(len(self.estimators_))
#                 for fold_counter, (tr_index, te_index) in enumerate(self.kf_.split(X, self._y_)) 
#                 )

#             S_train = self._unpack_parallel_preds(results)
#             return S_train

#         else:
#             results = Parallel(n_jobs=self.n_jobs)(delayed(_predict_model)(
#                     self.models_[estimator_counter][fold_counter], 
#                     X
#                 ) 
#                 for estimator_counter in range(len(self.estimators_))
#                 for fold_counter in range(self.n_folds)
#                 )

#             folds_preds_test = np.array([
#                 np.array(results[self.n_folds*i:self.n_folds*i+self.n_folds]) 
#                 for i in range(self.n_estimators_)
#                 ])

#             S_test = np.mean(folds_preds_test, axis=1).T

#             return S_test


#     def _random_choice(self, n, size, bound=2**30):
#         """
#         Random select size elements from a list 
#         of n ints without replacement
#         """
#         ids = [i for i in range(n)]        
#         return np.random.choice(ids, size=size, replace=False)


#     def _get_footprint(self, X, n_items=1000):
#         """Selects n_items random elements 
#         from 2d numpy array or matrix.
#         """
#         footprint = []
#         r, c = X.shape
#         n = r * c
#         ids = self._random_choice(n, min(n_items, n))

#         for i in ids:
#             row = i // c 
#             col = i - row * c
#             footprint.append((row, col, X[row, col]))

#         return footprint

     
#     def _check_identity(self, X,rtol=1e-05, atol=1e-08, equal_nan=False):
#         """Checks 2d numpy array or sparse matrix identity
#         by its shape and footprint.
#         """
#         if X.shape != self.train_shape_:
#             return False

#         return all([
#                 np.isclose(
#                 X[coo[0], coo[1]], 
#                 coo[2], rtol=rtol, 
#                 atol=atol, equal_nan=equal_nan
#                 ) for coo in self.train_footprint_
#             ])


#     def _get_params(self, attr, deep=True):
#         """Gives ability to get parameters of nested estimators
#         """
#         out = super(RegressionStackingTransformer, self).get_params(deep=False)
#         if not deep:
#             return out
#         estimators = getattr(self, attr)
#         if estimators is None:
#             return out
#         out.update(estimators)
#         for name, estimator in estimators:
#             for key, value in iter(estimator.get_params(deep=True).items()):
#                 out['%s__%s' % (name, key)] = value
#         return out


#     def get_params(self, deep=True):
#         return self._get_params('estimators', deep=deep)


#     def is_train_set(self, X):
#         """Checks if given data set was used to train.
#         """
#         check_is_fitted(self)
#         X = check_array(X, accept_sparse=['csr'], force_all_finite=True)
#         return self._check_identity(X)




