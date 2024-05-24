### Needed Imports
import pandas as pd
import numpy as np

import scipy.stats as stats

import sklearn.ensemble as skensemble
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score, mean_squared_error
import sklearn.model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

import xgboost as xgb

# Our Python package
import metabolinks.transformations as transf

import re
from tqdm import tqdm



## Functions related to annotation

def metabolite_annotation(annotated_data, dbs, ppm_margin, adduct_cols=[]):
    for d in dbs:
        print('Annotating with',d, end=' ')
        matched_ids_col = 'Matched '+d+' IDs'
        matched_names_col = 'Matched '+d+' names'
        matched_formulas_col = 'Matched '+d+' formulas'
        match_count_col = d+' match count'
        annotated_data[matched_ids_col] = ""
        annotated_data[matched_names_col] = ""
        annotated_data[matched_formulas_col] = ""
        annotated_data[match_count_col] = ""
        ##
        for a in tqdm(annotated_data.index):
            matched_ids = []
            matched_names = []
            matched_formulas = []
            mass_values = dbs[d]['DB'][dbs[d]['Mass_col']]
            ppm_dev = abs((mass_values-annotated_data['Neutral Mass'][a])/annotated_data[
                'Neutral Mass'][a])*10**6
            ppm_dev = ppm_dev[ppm_dev<ppm_margin] # ppm_margin used here
            for ad_col in adduct_cols:
                ppm_dev_ad = abs((mass_values-annotated_data[ad_col][a])/annotated_data[ad_col][a])*10**6
                ppm_dev_ad = ppm_dev_ad[ppm_dev_ad<ppm_margin]
                ppm_dev = pd.concat((ppm_dev, ppm_dev_ad))

            for i in ppm_dev.index:
                matched_ids.append(i)
                matched_names.append(dbs[d]['DB'][dbs[d]['Name_col']][i])
                matched_formulas.append(dbs[d]['DB'][dbs[d]['Formula_col']][i])

            if len(matched_ids) > 0:
                annotated_data.at[a, matched_ids_col] = matched_ids
                annotated_data.at[a, matched_names_col] = matched_names
                annotated_data.at[a, matched_formulas_col] = matched_formulas
                annotated_data.at[a, match_count_col] = len(matched_ids)
            else:
                annotated_data.at[a, matched_ids_col] = np.nan
                annotated_data.at[a, matched_names_col] = np.nan
                annotated_data.at[a, matched_formulas_col] = np.nan
                annotated_data.at[a, match_count_col] = np.nan
        print(f'-> Annotated {annotated_data[matched_ids_col].notnull().sum()} compounds')
        print('---------------')
    return annotated_data

# If your database doesn't have the monoisotopic masses of the compounds, but has the formulas, this will calculate them:
chemdict = {'H':1.007825031898,
            'C':12.000000000,
            'N':14.00307400425,
            'O':15.99491461926,
            'Na':22.98976928195,
            'P':30.97376199768,
            'S':31.97207117354,
            'Cl':34.968852694,
            'F':18.99840316207,
            'I':126.904473,
            'Ca':39.96259092,
            'Mg':23.985041709,
            'K':38.963706493,
            'Cr':49.9460413,
            'Co':58.9331943,
            'Cu':62.9295973,
            'Fe':55.9349362,
            'Al':26.98153843,
            'Mo':97.9054041,
            'Rb':84.911789743,
            'Mn':54.9380432,
            'Se':79.9165226,
            'Zr':89.90469888,
            'Ga':68.9255738,
            'Te':129.906222758,
            'Br':78.9183387,
            'Sn':119.9022026,
            'Ti':47.94794098,
            'W':183.9509335,
            'Si':27.9769265353,
            'Bi':208.980401,
            'As':74.9215956,
            'B':11.009305178,
            'Be':9.01218315,
            'Ni':57.9353423,
            'Ge':75.92140271,
            'V':50.9439573,
            'Ag':106.905092,
            'Hg':201.9706445,
            'Cd':113.9033652,
            'Sr':87.905612264,
            'Sb':120.903812,
            'Au':196.9665704,
            'Ba':137.9052472,
            'Ta':180.948001,
            'Pb':207.9766528,
            'Li':7.016003443,
            'Y':88.9058382,
            'Ru':101.9043403,
            'Cs':132.905451966,
            'Pd':105.9034808,
            'Pt':194.9647943,
            'Ce':139.905451,
            'La':138.906362,
            'Nd':141.907731,
            'Re':186.9557525,
            'Tl':204.9744278,
            'Gd':157.9241128,
            'Zn':63.9291425,
            'Hf':179.946561,
            'Th':232.038051,
            'He':4.00260325454,
            'Ar':39.962383122,
            'Nb':92.906371,
            'Sm':151.9197398,
            'Cm':243.061389322,
            'Eu':152.9212379,
            'Lu':174.9407778,
            'Sc':44.959074,
            'Fr':221.01425,
            'Pr':140.907661,
            'Tb':158.9253547,
            'Dy':163.9291815,
            'Ho':164.9303295,
            'Er':165.9302998,
            'Tm':168.9342195,
            'Os':191.961482,
            'Ir':192.9629249,
            'Ra':224.020203,
            'Tc':97.907219,
            'Xe':131.904160,
            'In':114.903877,
            'Kr':83.911507,
            'Ne':19.992439,
            'Ac':227.027740,
            'Bk':247.070297} 

def calculate_monoisotopic_mass(formula):
    """Returns the monoisotopic mass"""
        
    composition = element_composition(formula)
    
    mass = 0
    
    for e in composition:
        mass = mass + chemdict[e]*composition[e]
        
    
    return mass

elem_pattern = r'[A-Z][a-z]?\d*'
elem_groups = r'([A-Z][a-z]?)(\d*)'

def element_composition(formula, elements=None):
    """Given a string with a formula, return dictionary of element composition."""

    composition = {}
    for elemp in re.findall(elem_pattern, formula):
        match = re.match(elem_groups, elemp)
        n = match.group(2)
        number = int(n) if n != '' else 1
        composition[match.group(1)] = number

    if elements is None:
        return composition

    return {e : composition.get(e, 0) for e in elements}




### Functions related to Data Pre-treatment

def basic_feat_filtering(file, target=None, filt_method='total_samples', filt_kw=2,
                  extra_filt=None, extra_filt_data=None):
    "Performs feature filtering in 2 steps."
    
    # Filtering based on the sampels each feature appears in
    if filt_method == 'total_samples': # Filter features based on the times (filt_kw) they appear in the dataset
        # Minimum can be a percentage if it is a value between 0 and 1!
        data_filt = transf.keep_atleast(file, minimum=filt_kw)
    elif filt_method == 'class_samples': # Features retained if they appear filt_kw times in the samples of at least one class
        # Minimum can be a percentage if it is a value between 0 and 1!
        data_filt = transf.keep_atleast(file, minimum=filt_kw, y=np.array(target))
    elif filt_method == None: # No filtering
        data_filt = file.copy()
    else:
        raise ValueError('Feature Filtering strategy not accepted/implemented in function. Implement if new strategy.')
        
    # Extra filtering based if the features are annotated
    if extra_filt == 'Formula': # Keep only features with a formula annotated on the dataset
        meta_cols_formulas = [i for i in extra_filt_data.columns if 'formulas' in i]
        if 'Formula' in extra_filt_data.columns:
            idxs_to_keep = [i for i in data_filt.columns if type(extra_filt_data.loc[i, 'Formula']) == str]
        else:
            idxs_to_keep = []
        for col in meta_cols_formulas:
            idxs_to_keep.extend([i for i in data_filt.columns if type(extra_filt_data.loc[i, col]) == list])
        idxs_to_keep = pd.unique(np.array(idxs_to_keep))
        data_filt = data_filt.loc[:,idxs_to_keep]

    elif extra_filt == 'Name': # Keep only features with a name annotated on the dataset
        meta_cols_names = [i for i in extra_filt_data.columns if 'names' in i]
        if 'Name' in extra_filt_data.columns:
            idxs_to_keep = [i for i in data_filt.columns if type(extra_filt_data.loc[i, 'Name']) == str]
        else:
            idxs_to_keep = []
        for col in meta_cols_names:
            idxs_to_keep.extend([i for i in data_filt.columns if type(extra_filt_data.loc[i, col]) == list])
        idxs_to_keep = pd.unique(np.array(idxs_to_keep))
        data_filt = data_filt.loc[:,idxs_to_keep]

    elif extra_filt == None: # No extra filtering
        data_filt = data_filt.copy()
    else:
        raise ValueError('Feature Filtering strategy not accepted/implemented in function. Implement if new strategy.')
    
    return data_filt

def missing_value_imputer(data_filt, mvi='min_sample', mvi_kw=1/5):
    "Performs Missing Value Imputation of choice based on parameters passed."
    
    # Missing Value Imputation
    if mvi == 'min_sample': # Replace NaN's by a fraction (mvi_kw) of the minimum value of the sample the NaN belongs to.
        imputed = transf.fillna_frac_min_feature(data_filt.T, fraction=mvi_kw).T
    elif mvi == 'min_feat': # Replace NaN's by a fraction (mvi_kw) of the minimum value of the feature the NaN belongs to.
        imputed = transf.fillna_frac_min_feature(data_filt, fraction=mvi_kw)
    elif mvi == 'min_data': # Replace NaN's by a fraction (mvi_kw) of the minimum value in the dataset.
        imputed = transf.fillna_frac_min(data_filt, fraction=mvi_kw)
    elif mvi == 'zero': # Replace NaN's by zero (no mvi_kw).
        imputed = transf.fillna_zero(data_filt)
    else:
        raise ValueError('Missing Value Imputation strategy not accepted/implemented in function. Implement if new strategy.')
        
    return imputed

def normalizer(data, norm='ref_feat', norm_kw='555.2692975341 Da'):
    "Performs Normalization of choice based on parameters passed."
        
    # Normalizations
    if norm == 'ref_feat': # Normalization by a reference feature indicated by the norm_kw
        N = (data.T/data.loc[:, norm_kw])
        N = N.drop(norm_kw)
        N = N.T
        #N = transf.normalize_ref_feature(data, feature=norm_kw, remove=True)
    elif norm == 'total_sum': # Normalization by the total sum of intensities (no norm_kw)
        N = transf.normalize_sum(data)
    elif norm == 'PQN': # Normalization by Probabilistic Quotient Normalization (norm_kw is ref_sample, usually, 'mean')
        N = transf.normalize_PQN(data, ref_sample=norm_kw)
    elif norm == 'Quantile': # Normalization by Quantile Normalization (norm_kw is ref_feat, usually, 'mean')
        N = transf.normalize_quantile(data, ref_type=norm_kw)
    elif norm == None: # No Normalization
        N = data.copy()
    else:
        raise ValueError('Normalization strategy not accepted/implemented in function. Implement if new strategy.')
    
    return N

def transformer(N, tf='glog', tf_kw=None):
    "Performs Transformation of choice based on parameters passed."
        
    # Transformations
    if tf == 'glog': # Generalized Logarithmic Transformation with lambda = trans_kw, usually, None.
        NG = transf.glog(N, lamb=tf_kw)
    elif tf == None: # No Transformation
        NG = N.copy()
    else:
        raise ValueError('Transforamtion strategy not accepted/implemented in function. Implement if new strategy.')
    
    return NG

def scaler(NG, scaling='pareto', scaling_kw=None):
    "Performs Scaling of choice based on parameters passed."
    
    # Scalings
    if scaling == 'pareto': # Pareto scaling (no scaling_kw)
        NGP = transf.pareto_scale(NG)
    elif scaling == 'mean_center': # Just mean centering, no scaling (no scaling_kw)
        NGP = transf.mean_center(NG)
    elif scaling == 'auto': # Auto or Standard Scaling (no scaling_kw)
        NGP = transf.auto_scale(NG)
    elif scaling == 'range': # Range Scaling (no scaling_kw)
        NGP = transf.range_scale(NG)
    elif scaling == 'vast': # Vast Scaling (no scaling_kw)
        NGP = transf.vast_scale(NG)
    elif scaling == 'level': # Level Scaling (scaling_kw is boolean - True or False if average, usually False)
        NGP = transf.level_scale(NG, average=scaling_kw)
    elif scaling == None: # No Scaling
        NGP = NG.copy()
    else:
        raise ValueError('Scaling strategy not accepted/implemented in function. Implement if new strategy.')
    
    return NGP

def filtering_pretreatment(data, target, sample_cols,
                  filt_method='total_samples', filt_kw=2, # Filtering based on number of times features appear
                  extra_filt=None, # Filtering based on annotation of features ('Formula' or 'Name')
                  mvi='min_sample', mvi_kw=1/5, # Missing value imputation
                  norm='ref_feat', norm_kw='555.2692975341 Da', # Normalization
                  tf='glog', tf_kw=None, # Transformation
                  scaling='pareto', scaling_kw=None): # Scaling
    """Performs all feature filtering and data pre-treatments of choice based on parameters passed.
    
       Returns: Five DataFrames."""
    
    # Cols for the meta data and with the samples
    sample_cols = sample_cols
    meta_cols = [i for i in data.columns if i not in sample_cols]
    
    # Separates feature intensity data from "metadata" (m/z and annotations)
    meta_data = data[meta_cols]
    sample_data = data[sample_cols].T
    
    # Filtering
    filt_sample_data = basic_feat_filtering(sample_data, target, filt_method=filt_method, filt_kw=filt_kw,
                                            extra_filt=extra_filt, extra_filt_data=meta_data)
    
    # Treated data DataFrame
    imputed = missing_value_imputer(filt_sample_data.copy(), mvi=mvi, mvi_kw=mvi_kw) # Missing Value Imputation
    N = normalizer(imputed, norm=norm, norm_kw=norm_kw) # Normalization
    NG = transformer(N, tf=tf, tf_kw=tf_kw) # Transformation
    NGP = scaler(NG, scaling=scaling, scaling_kw=scaling_kw) # Scaling
    
    # Meta data DataFrame
    meta_data = meta_data.reindex(NGP.columns)

    # processed_data DataFrame
    norm_sample_data=normalizer(filt_sample_data, norm=norm, norm_kw=norm_kw) # Normalization without missing value imputation
    inverted_norm_sample_data = norm_sample_data.T
    processed_data = pd.concat([meta_data, inverted_norm_sample_data], axis=1,join='inner') # Obtaining the processed dataframe
    
    # univariate_data DataFrame
    univariate_data = N.copy()
    
    # BinSim DataFrame
    BinSim = filt_sample_data.mask(filt_sample_data.notnull(), 1).mask(filt_sample_data.isnull(), 0)
    if norm == 'ref_feat':
        BinSim = BinSim.drop(columns=norm_kw)
    
    return NGP, processed_data, univariate_data, meta_data, BinSim



### Functions related to Common and Exclusive Metabolites

def common(samples):
    """Given a list of n samples, compute common features (intersection).
    
       Returns a DataFrame with common features"""
    
    df = pd.concat(samples, axis=1, join='inner', keys=range(len(samples)))
    return df[0]

def exclusive(samples):
    """Given a list of samples, compute exclusive features for each sample.
    
       Returns a list of DataFrames with exclusive features for each corresponding sample in input"""
    
    # concat all samples
    concatenation = pd.concat(samples)
    
    # find indexes that occur only once
    reps = concatenation.index.value_counts()
    exclusive_feature_counts = reps[reps == 1]
    
    # keep only those in each sample

    exclusive = [s[s.index.isin(exclusive_feature_counts.index)] for s in samples]
    return exclusive



### Functions related to perform PCA (unsupervised analysis)

def compute_df_with_PCs_VE_loadings(df, n_components=5, whiten=True, labels=None, return_var_ratios_and_loadings=False):
    pca = PCA(n_components=n_components, svd_solver='full', whiten=whiten)
    pc_coords = pca.fit_transform(df)
    var_explained = pca.explained_variance_ratio_[:pca.n_components_]
    loadings = pca.components_[:pca.n_components_].T

    # concat labels to PCA coords (in a DataFrame)
    principaldf = pd.DataFrame(pc_coords, index=df.index, columns=[f'PC {i}' for i in range(1, pca.n_components_+1)])
    if labels is not None:
        labels_col = pd.DataFrame(labels, index=principaldf.index, columns=['Label'])
        principaldf = pd.concat([principaldf, labels_col], axis=1)
    if not return_var_ratios_and_loadings:
        return principaldf
    else:
        return principaldf, var_explained, loadings
    



### Functions related to Random Forests

def RF_model(df, y, regres=False, return_cv=True, iter_num=1, n_trees=200, cv=None, n_fold=5,
             metrics = ('accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'), **kwargs):
    "Fitting RF models and rturning the models and their cross-validation scores."
    results = {}

    if regres:
        fitted_model = skensemble.RandomForestRegressor(n_estimators=n_trees)
    else:
        fitted_model = skensemble.RandomForestClassifier(n_estimators=n_trees)
    
    fitted_model = fitted_model.fit(df, y)
    results['model'] = fitted_model

    # Setting up variables for imp_feat storing
    imp_feat = np.zeros((iter_num * n_fold, len(df.columns)))
    f = 0

    if not return_cv:
        return(fitted_model)
    if cv is None:
        cv = sklearn.model_selection.StratifiedKFold(n_fold, shuffle=True)

    store_res = {m:[] for m in metrics}

    for _ in range(iter_num):
        if regres:
            rf = skensemble.RandomForestRegressor(n_estimators=n_trees)
        else:
            rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
        
        cv_res = sklearn.model_selection.cross_validate(rf, df, y, cv=cv, scoring=metrics, **kwargs)
        for i in metrics:
            store_res[i].extend(cv_res['test_'+i])

        for train_index, test_index in cv.split(df, y):
            # Random Forest setup and fit
            if regres:
                rf = skensemble.RandomForestRegressor(n_estimators=n_trees)
            else:
                rf = skensemble.RandomForestClassifier(n_estimators=n_trees)
            X_train, X_test = df.iloc[train_index, :], df.iloc[test_index, :]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            rf.fit(X_train, y_train)

            # Compute important features
            imp_feat[f, :] = rf.feature_importances_ # Importance of each feature
            f = f + 1

    # Collect and order all important features values from each Random Forest
    imp_feat_sum = imp_feat.sum(axis=0) / (iter_num * n_fold)
    results['imp_feat'] = sorted(enumerate(imp_feat_sum), key=lambda x: x[1], reverse=True)

    results.update(store_res)
    return results#{'model': fitted_model, 'cv_scores': scores}

# Metabolite statistic

def metabolite_statistics(finder, groups):
    gfinder = finder.copy()
    for g in groups:
        gfinder[g+' Average'] = gfinder[gfinder.columns.intersection(groups[g])].mean(axis=1)
        gfinder[g+' std'] = gfinder[gfinder.columns.intersection(groups[g])].std(axis=1)
        
    g_list = list(groups)


    t_test = stats.ttest_ind(gfinder[gfinder.columns.intersection(groups[g_list[0]])], 
                                            gfinder[gfinder.columns.intersection(groups[g_list[1]])], axis=1)

    gfinder['T-test Statistic'] = t_test.statistic

    gfinder['T-test p-value'] = t_test.pvalue

        
    avg_cols = [col for col in gfinder.columns if 'Average' in col]
    std_cols = [col for col in gfinder.columns if 'std' in col]
    t_test_cols = ['T-test Statistic', 'T-test p-value']

    gfinder = gfinder[avg_cols+std_cols+t_test_cols]

    return gfinder
