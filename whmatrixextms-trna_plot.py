#!/usr/bin/env python

#######################################################################
## IMPORTS ##
#######################################################################

import os
import argparse
import pathlib
from pathlib import Path

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import itertools
import math
import time
from sklearn.model_selection import train_test_split
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import copy

#######################################################################
## CLASSES ##
#######################################################################

class SparseModel:
    """
    A class for the storage of genotype-phenotype data from a DMS experiment.
    """
    def __init__(
        self, 
        vtable,
        max_interaction_order = 1,
        ensemble = False):
        """
        Initialize a DMSData object.

        :param vtable: Pandas DataFrame with DMS data (required).
        :param max_interaction_order: Maximum interaction order (default:1).
        :param ensemble: Ensemble encode features. (default:False).
        :returns: MochiData object.
        """
        #Save attributes
        self.vtable = vtable
        self.max_interaction_order = max_interaction_order
        self.ensemble = ensemble
        #Initialize attributes
        self.variantCol = 'nt_seq'
        self.wildtype_split = None
        self.fitness = None
        self.Xoh = None
        self.Xohi = None
        self.feature_names = None
        self.models = {}

        #Input DMS data
        self.vtable = vtable
        #Fitness
        self.fitness = self.vtable.loc[:,['fitness', 'sigma']]
        #Wild-type sequence
        self.wildtype = str(np.array(self.vtable.loc[self.vtable['WT'] == True, self.variantCol])[0])
        self.wildtype_split = [c for c in self.wildtype]
        #Sequence features
        self.X = self.vtable[self.variantCol].str.split('', expand=True).iloc[:, 1:-1]
        #One hot encode sequence features
        print("One-hot encoding sequence features")
        self.Xoh = self.one_hot_encode_features()
        #One-hot encode interaction features
        print("One-hot encoding interaction features")
        self.one_hot_encode_interactions(
            max_order = self.max_interaction_order)
        #Ensemble encode features
        if self.ensemble:
            print("Ensemble encoding features")
            self.Xohi = self.ensemble_encode_features()
        print("Done!")

    def combinations(
        self,
        n,
        k):
        """
        The number of k-combinations from a set of n elements.

        :param n: Set size. (required).
        :param k: Number of distinct elements from set. (required).
        :returns: Number of k-combinations.
        """
        return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

    def pairwise_interaction_enrichment(
        self,
        input_model,
        interactions = [["1","10"], ["2","9"], ["3","7"]],
        coefficient_order = [2]):
        """
        Enrichment of non-zero coefficients for pairwise interactions.

        :param input_model: Sparse model. (required).
        :param interactions: List of lists of pairwise interactions to test. (default: [["1","10"], ["2","9"], ["3","7"]]).
        :param coefficient_order: List of sparse model coefficient orders to test. (default: [2]).
        :returns: Dictionary of Fisher's Exact Test results.
        """
        #Zero and non-zero features
        all_feat = input_model.feature_names_in_
        z_feat = [re.sub("[acgt]", "", all_feat[i]).split("_") for i in range(len(input_model.coef_)) if input_model.coef_[i]==0]
        z_feat = [i for i in z_feat if len(i) in coefficient_order]
        nz_feat = [re.sub("[acgt]", "", all_feat[i]).split("_") for i in range(len(input_model.coef_)) if input_model.coef_[i]!=0]
        nz_feat = [i for i in nz_feat if len(i) in coefficient_order]
        #Number of interactions in zero features
        z_feat_int = 0
        nz_feat_int = 0
        for i in interactions:
            z_feat_int += len([j for j in z_feat if i[0] in j and i[1] in j])
            nz_feat_int += len([j for j in nz_feat if i[0] in j and i[1] in j])
        #2x2 contingency table
        cont_table = np.array([[nz_feat_int,
                         z_feat_int], 
                        [sum([self.combinations(len(i), 2) for i in nz_feat])-nz_feat_int,
                         sum([self.combinations(len(i), 2) for i in z_feat])-z_feat_int]])
        #Perform Fishers Exact Test
        odds_ratio, p_value = stats.fisher_exact(cont_table)
        return {'odd_ratio': odds_ratio, 'p_value': p_value, 'cont_table': cont_table}
        
    def fit_lasso(
        self,
        name,
        l1_lambdas = arange(0.005, 0.5, 0.005),
        test_size = 0.1,
        random_state = 42):
        """
        Fit Lasso regression model.

        :param name: Model name. (required).
        :param l1_lambdas: List or array of L1 lambdas. (default:arange(0.005, 0.5, 0.005)).
        :param test_size: Proportion of samples to use for test set. (default:arange(0.005, 0.5, 0.005)).
        :param random_state: Random state for training/test split. (default:42).
        :returns: A DataFrame with 1-hot sequences.
        """
 
        X = self.Xohi
        y = self.fitness['fitness']

        #Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        #Define cross-validation method to evaluate model
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        #Define model
        self.models[name] = LassoCV(alphas=l1_lambdas, cv=cv, n_jobs=-1, fit_intercept = False)

        #Fit model
        self.models[name].fit(X_train, y_train)

        #Lambda that produced the lowest test MSE
        self.models[name].best_lambda = self.models[name].alpha_ 

        #Performance on test data
        y_test_hat = self.models[name].predict(X_test)
        self.models[name].test_corr = np.corrcoef(y_test, y_test_hat)[0,1]
        
        #Number of non-zero coefficients
        self.models[name].coef_nonzero = len([i for i in self.models[name].coef_ if i!=0])
        
        #Enrichment for pairwise interactions
        self.models[name].interaction_enrich = self.pairwise_interaction_enrichment(
            input_model = self.models[name],
            coefficient_order = list(range(2, self.max_interaction_order+1)))
        
        print("Done!")

    def one_hot_encode_features(
        self,
        include_WT = True):
        """
        1-hot encode sequences.

        :param include_WT: Whether or not to include WT feature (default:True).
        :returns: A DataFrame with 1-hot sequences.
        """
        enc = OneHotEncoder(
            handle_unknown='ignore', 
            drop = np.array(self.wildtype_split), 
            dtype = int)
        enc.fit(self.X)
        one_hot_names = [self.wildtype_split[int(i[1:-2])]+str(int(i[1:-2])+1)+i[-1] for i in enc.get_feature_names_out()]
        one_hot_df = pd.DataFrame(enc.transform(self.X).toarray(), columns = one_hot_names)
        if include_WT:
            one_hot_df = pd.concat([pd.DataFrame({'WT': [1]*len(one_hot_df)}), one_hot_df], axis=1)
        return one_hot_df

    def merge_dictionaries(
        self,
        dict1,
        dict2):
        """
        Merge two dictionaries by keys (and unique values).

        :param dict1: Dictionary 1.
        :param dict2: Dictionary 2.
        :returns: merged dictionary.
        """
        for k in dict2:
            if k in dict1:
                dict1[k] = list(set(dict1[k] + dict2[k]))
            else:
                dict1[k] = dict2[k]
        return dict1

    def get_theoretical_interactions_phenotype(
        self, 
        max_order = 2):
        """
        Get theoretical interaction features for variants corresponding to a specific phenotype.

        :param max_order: Maximum interaction order (default:2).
        :returns: dictionary of interaction features.
        """
        #Mutations observed for this phenotype
        mut_count = list(self.Xoh.sum(axis = 0))
        pheno_mut = [self.Xoh.columns[i] for i in range(len(self.Xoh.columns)) if mut_count[i]!=0]
        #All possible combinations of mutations
        all_pos = list(set([i[1:-1] for i in pheno_mut if i!="WT"]))
        all_pos_mut = {int(i):[j for j in pheno_mut if j[1:-1]==i] for i in all_pos}
        all_features = {}
        for n in range(max_order + 1):
            #Order at least 2
            if n>1:
                all_features[n] = []
                #All position combinations
                pos_comb = list(itertools.combinations(sorted(all_pos_mut.keys()), n))
                for p in pos_comb:
                    #All mutation combinations for these positions
                    all_features[n] += ["_".join(c) for c in itertools.product(*[all_pos_mut[j] for j in p])]
        return all_features

    def get_theoretical_interactions(
        self, 
        max_order = 2):
        """
        Get theoretical interaction features.

        :param max_order: Maximum interaction order (default:2).
        :returns: tuple with dictionary of interaction features and dictionary of order counts.
        """
        #All features
        all_features = self.get_theoretical_interactions_phenotype(max_order = max_order)
        #All feature orders
        int_order_dict = {k:len(all_features[k]) for k in all_features}
        return (all_features, int_order_dict)

    def one_hot_encode_interactions(
        self, 
        max_order = 2,
        max_cells = 1e9):
        """
        Add interaction terms to 1-hot encoding DataFrame.

        :param max_order: Maximum interaction order (default:2).
        :param max_cells: Maximum matrix cells permitted (default:1billion).
        :returns: Nothing.
        """
        
        #Check if no interactions to add
        if max_order<2:
            self.Xohi = copy.deepcopy(self.Xoh)
            return

        #Get all theoretical interactions
        all_features,int_order_dict = self.get_theoretical_interactions(max_order = max_order)
        print("... Total theoretical features (order:count): "+", ".join([str(i)+":"+str(int_order_dict[i]) for i in sorted(int_order_dict.keys())]))
        #Flatten
        all_features_flat = list(itertools.chain(*list(all_features.values())))

        #Select interactions
        int_list = []
        int_order_dict_retained = {}
        int_list_names = []
        all_features_loop = {0: all_features_flat}

        #Loop over all orders
        for n in all_features_loop.keys():
            #Loop over all features of this order
            for c in all_features_loop[n]:
                c_split = c.split("_")
                int_col = (self.Xoh.loc[:,c_split].sum(axis = 1)==len(c_split)).astype(int)
                #Check if minimum number of observations satisfied
                if sum(int_col) >= 0:
                    int_list += [int_col]
                    int_list_names += [c]
                    if len(c_split) not in int_order_dict_retained.keys():
                        int_order_dict_retained[len(c_split)] = 1
                    else:
                        int_order_dict_retained[len(c_split)] += 1
                #Check memory footprint
                if len(int_list)*len(self.Xoh) > max_cells:
                    print(f"Error: Too many interaction terms: number of feature matrix cells >{max_cells:>.0e}")
                    raise ValueError

        print("... Total retained features (order:count): "+", ".join([str(i)+":"+str(int_order_dict_retained[i])+" ("+str(round(int_order_dict_retained[i]/int_order_dict[i]*100, 1))+"%)" for i in sorted(int_order_dict_retained.keys())]))

        #Concatenate into dataframe
        if len(int_list)>0:
            self.Xohi = pd.concat(int_list, axis=1)
            self.Xohi.columns = int_list_names
            #Reorder
            self.Xohi = self.Xohi.loc[:,[i for i in all_features_flat if i in self.Xohi.columns]]
            self.Xohi = pd.concat([self.Xoh, self.Xohi], axis=1)
        else:
            self.Xohi = copy.deepcopy(self.Xoh)

        #Save interaction feature names
        self.feature_names = self.Xohi.columns

    def H_matrix(
        self,
        str_geno,
        str_coef,
        num_states = 2,
        invert = False):
        """
        Construct Walsh-Hadamard matrix.

        :param str_geno: list of genotype strings where '0' indicates WT state.
        :param str_coef: list of coefficient strings where '0' indicates WT state.
        :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences.
        :param invert: invert the matrix.
        :returns: Walsh-Hadamard matrix as a numpy matrix.
        """
        #Genotype string length
        string_length = len(str_geno[0])
        #Number of states per position in genotype string (float)
        if type(num_states) == int:
            num_states = [float(num_states) for i in range(string_length)]
        else:
            num_states = [float(i) for i in num_states]
        #Convert reference characters to "." and binary encode
        str_coef = [[ord(j) for j in i.replace("0", ".")] for i in str_coef]
        str_geno = [[ord(j) for j in i] for i in str_geno]
        #Matrix representations
        num_statesi = np.repeat([num_states], len(str_geno)*len(str_coef), axis = 0)
        str_genobi = np.repeat(str_geno, len(str_coef), axis = 0)
        str_coefbi = np.transpose(np.tile(np.transpose(np.asarray(str_coef)), len(str_geno)))
        str_genobi_eq_str_coefbi = (str_genobi == str_coefbi)
        #Factors
        row_factor2 = str_genobi_eq_str_coefbi.sum(axis = 1)
        if invert:
            row_factor1 = np.prod(str_genobi_eq_str_coefbi * (num_statesi-2) + 1, axis = 1)       
            return ((row_factor1 * np.power(-1, row_factor2))/np.prod(num_states)).reshape((len(str_geno),-1))
        else:
            row_factor1 = (np.logical_or(np.logical_or(str_genobi_eq_str_coefbi, str_genobi==ord('0')), str_coefbi==ord('.')).sum(axis = 1) == string_length).astype(float)            
            return ((row_factor1 * np.power(-1, row_factor2))).reshape((len(str_geno),-1))

    def H_matrix_chunker(
        self,
        str_geno,
        str_coef,
        num_states = 2,
        invert = False,
        chunk_size = 1000):
        """
        Construct Walsh-Hadamard matrix in chunks.

        :param str_geno: list of genotype strings where '0' indicates WT state.
        :param str_coef: list of coefficient strings where '0' indicates WT state.
        :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences.
        :param invert: invert the matrix.
        :param chunk_size: chunk size in number of genotypes/variants (default:1000).
        :returns: Walsh-Hadamard matrix as a numpy matrix.
        """

        #Check if chunking not necessary
        if len(str_geno) < chunk_size:
            return self.H_matrix(
                str_geno = str_geno, 
                str_coef = str_coef, 
                num_states = num_states, 
                invert = invert)

        #Chunk
        hmat_list = []
        for i in range(math.ceil(len(str_geno)/chunk_size)):
            from_i = (i*chunk_size)
            to_i = (i+1)*chunk_size
            if to_i > len(str_geno):
                to_i = len(str_geno)
            hmat_list += [self.H_matrix(
                str_geno = str_geno[from_i:to_i], 
                str_coef = str_coef, 
                num_states = num_states, 
                invert = invert)]
        return np.concatenate(hmat_list, axis = 0)

    def V_matrix(
        self,
        str_coef,
        num_states = 2,
        invert = False):
        """
        Construct diagonal weighting matrix.

        :param str_coef: list of coefficient strings where '0' indicates WT state.
        :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences.
        :param invert: invert the matrix.
        :returns: diagonal weighting matrix as a numpy matrix.
        """
        #Genotype subset
        str_geno = str_coef
        #Genotype string length
        string_length = len(str_geno[0])
        #Number of states per position in genotype string
        if type(num_states) == int:
            num_states = [float(num_states) for i in range(string_length)]
        else:
            num_states = [float(i) for i in num_states]
        #Convert reference characters to "."
        str_coef_ = [i.replace("0", ".") for i in str_coef]
        #initialize V matrix
        V = np.array([[0.0]*len(str_coef)]*len(str_geno))
        #Fill matrix
        for i in range(len(str_geno)):
            factor1 = int(np.prod([c for a,b,c in zip(str_coef_[i], str_geno[i], num_states) if ord(a) != ord(b)]))
            factor2 = sum([1 for a,b in zip(str_coef_[i], str_geno[i]) if ord(a) == ord(b)])
            if invert:
                V[i,i] = factor1 * np.power(-1, factor2)
            else:
                V[i,i] = 1/(factor1 * np.power(-1, factor2))
        return V

    def coefficient_to_sequence(
        self,
        coefficient,
        length):
        """
        Get sequence representation of a coefficient string.

        :param coefficient: coefficient string.
        :param length: integer sequence length.
        :returns: sequence string.
        """
        #initialize sequence string
        coefficient_seq = ['0']*length
        #Wild-type sequence string
        if coefficient == "WT":
            return ''.join(coefficient_seq)
        #Variant sequence string
        for i in coefficient.split("_"):
            coefficient_seq[int(i[1:-1])-1] = i[-1]
        return ''.join(coefficient_seq)

    def ensemble_encode_features(
        self):
        """
        Ensemble encode features.

        :returns: Nothing.
        """
        #Wild-type mask variant sequences
        geno_list = list(self.vtable.apply(lambda row : "".join(x if x!=y else '0' for x,y in zip(str(row[self.variantCol]),self.wildtype)),
            axis = 1))
        #Sequence representation of 1-hot encoded coefficients/features
        ceof_list = [self.coefficient_to_sequence(coef, len(self.wildtype)) for coef in self.Xohi.columns]
        #Number of states per position
        state_list = (self.X.apply(lambda column: column.value_counts(), axis = 0)>0).apply(lambda column: column.value_counts(), axis = 0)
        state_list = list(np.asarray(state_list)[0])
        #Ensemble encode features
        start = time.time()
        hmat_inv = self.H_matrix_chunker(
            str_geno = geno_list, 
            str_coef = ceof_list, 
            num_states = state_list, 
            invert = True)
        end = time.time()
        print("Construction time for H_matrix :", end-start)
        vmat_inv = self.V_matrix(
            str_coef = ceof_list, 
            num_states = state_list, 
            invert = True)
        return pd.DataFrame(np.matmul(hmat_inv, vmat_inv), columns = self.Xohi.columns)

#######################################################################
## FUNCTIONS ##
#######################################################################

def init_argparse(
    demo_mode = False
    ) -> argparse.ArgumentParser:
    """
    Initialize command line argument parser.

    :returns: ArgumentParser.
    """

    parser = argparse.ArgumentParser(
        description="Command Line tool."
    )
    parser.add_argument('--fitness_file', type = pathlib.Path, default = ".", help = "fitness file")
    parser.add_argument('--output_dir', type = pathlib.Path, default = ".", help = "output directory")
    parser.add_argument('--plot_dir', type = pathlib.Path, default = ".", help = "plot directory")
    return parser

def custom_linear_colour_palette(
    colour_list = ['white', 'black'],
    n_colours = 2,
    start = 0.0,
    stop = 1.0):
    """
    Custom linear segmented colour palette.

    :param colour_list: List of colours. (required).
    :param n_colours: Number of colours for palette. (required).
    :param start: Float starting colour in the range [0.0,1.0]. (default:0.0).
    :param stop: Float stop colour in the range [0.0,1.0]. (default:0.0).
    :returns: List of colours.
    """
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('mycols', colour_list)
    return [matplotlib.colors.to_hex(color_map(i/256.0)) for i in list(range(int(start * 256), int(stop * 256)+1, int((stop-start)/(n_colours-1) * 256)))]

def combinations(
    n,
    k):
    """
    The number of k-combinations from a set of n elements.

    :param n: Set size. (required).
    :param k: Number of distinct elements from set. (required).
    :returns: Number of k-combinations.
    """
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

def pairwise_interaction_enrichment_null(
    input_model,
    interactions = [["1","10"], ["2","9"], ["3","7"]],
    coefficient_order = [2],
    seed = 1,
    int_feat_dict = {}):
    """
    Enrichment of non-zero coefficients for pairwise interactions.

    :param input_model: Sparse model. (required).
    :param interactions: List of lists of pairwise interactions to test. (default: [["1","10"], ["2","9"], ["3","7"]]).
    :param coefficient_order: List of sparse model coefficient orders to test. (default: [2]).
    :param seed: Random seed for sampling coefficients. (default:1).
    :param int_feat_dict: Dictionary of all interaction features per order. (default:{}).
    :returns: Dictionary of Fisher's Exact Test results.
    """
    #Zero and non-zero features
    all_feat = input_model.feature_names_in_
    nz_feat = [re.sub("[acgt]", "", all_feat[i]).split("_") for i in range(len(input_model.coef_)) if input_model.coef_[i]!=0]
    nz_feat = [i for i in nz_feat if len(i) in coefficient_order]
    #Number of non-zero coefficients of each order
    nz_feat_norder = dict(pd.DataFrame({'coef_order': np.asarray([len(i) for i in nz_feat if len(i)])}).value_counts('coef_order'))
    #Randomly sample same number of coefficients of same order
    nz_feat = []
    np.random.seed(seed)
    for co in nz_feat_norder.keys():
        all_feat_order = int_feat_dict[co]
        nz_feat += list(np.random.choice(all_feat_order, nz_feat_norder[co], replace=False))
    z_feat = [re.sub("[acgt]", "", i).split("_") for i in all_feat if not i in nz_feat]
    z_feat = [i for i in z_feat if len(i) in coefficient_order]
    nz_feat = [re.sub("[acgt]", "", i).split("_") for i in nz_feat]
    #Number of interactions in zero features
    z_feat_int = 0
    nz_feat_int = 0
    for i in interactions:
        z_feat_int += len([j for j in z_feat if i[0] in j and i[1] in j])
        nz_feat_int += len([j for j in nz_feat if i[0] in j and i[1] in j])
    #2x2 contingency table
    cont_table = np.array([[nz_feat_int,
                     z_feat_int], 
                    [sum([combinations(len(i), 2) for i in nz_feat])-nz_feat_int,
                     sum([combinations(len(i), 2) for i in z_feat])-z_feat_int]])
    #Perform Fishers Exact Test
    odds_ratio, p_value = stats.fisher_exact(cont_table)
    return {'odd_ratio': odds_ratio, 'p_value': p_value, 'cont_table': cont_table}

#######################################################################
## MAIN ##
#######################################################################

def main():
    """
    Main function.

    :returns: Nothing.
    """

    #Get command line arguments
    parser = init_argparse()
    args = parser.parse_args()

    #Globals
    # fitness_file="/users/project/prj004631/afaure/DMS/dimsumrun_JD_Phylogeny_tR-R-CCU/JD_Phylogeny_tR-R-CCU_dimsum1.3/JD_Phylogeny_tR-R-CCU_dimsum1.3_fitness_replicates.txt"
    # output_dir = "/users/project/prj004631/afaure/DMS/Results/epistasis_ms/output"
    # plot_dir = "/users/project/prj004631/afaure/DMS/Results/epistasis_ms/plots"
    fitness_file = args.fitness_file
    output_dir = args.output_dir
    plot_dir = args.plot_dir

    #Make output dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    #Load dimsum data
    fitness_df = pd.read_csv(fitness_file, sep = "\t")

    #Filter fitness data...
    #Fitness defined for all biological replicates and at least 10 read counts in all input samples
    fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness1_uncorr'])==False) & (fitness_df['count_e1_s0']>=10),:]
    fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness2_uncorr'])==False) & (fitness_df['count_e2_s0']>=10),:]
    fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness3_uncorr'])==False) & (fitness_df['count_e3_s0']>=10),:]
    fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness4_uncorr'])==False) & (fitness_df['count_e4_s0']>=10),:]
    fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness5_uncorr'])==False) & (fitness_df['count_e5_s0']>=10),:]
    fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness6_uncorr'])==False) & (fitness_df['count_e6_s0']>=10),:]

    #Percentage explainable variance
    fitness_mat = np.array(fitness_df[['fitness1_uncorr', 'fitness2_uncorr', 'fitness3_uncorr', 'fitness4_uncorr', 'fitness5_uncorr', 'fitness6_uncorr']])
    #Correlation between all replicates
    fitness_corr = np.corrcoef(np.transpose(fitness_mat))[np.triu_indices(6, k = 1)]
    #Percentage explainable variance
    var_expl = np.power(np.mean(fitness_corr), 2)

    #######################################################################
    ## PLOT DATA ##
    #######################################################################

    df_list = []
    int_feat_dict = {}
    for seed in range(1, 100+1):
        print(seed)
        #Load
        with open(os.path.join(output_dir, 'lasso_'+'ensemble'+"_"+str(seed)+'.pkl'), 'rb') as inp:
            lasso_e = pickle.load(inp)
        with open(os.path.join(output_dir, 'lasso_'+'1hot'+"_"+str(seed)+'.pkl'), 'rb') as inp:
            lasso_1 = pickle.load(inp)

        #All features per order 
        if int_feat_dict == {}:
            for co in list(range(2, lasso_1.max_interaction_order+1)):
                int_feat_dict[co] = [i for i in lasso_1.feature_names if len(re.sub("[acgt]", "", i).split("_"))==co]

        #1-hot
        coef_nzero = [[lasso_1.models[m].feature_names_in_[i] for i in range(len(lasso_1.models[m].feature_names_in_)) if lasso_1.models[m].coef_[i]!=0] for m in lasso_1.models.keys()]
        coef_nzero_int = [[j for j in i if len(j.split("_"))>=2] for i in coef_nzero]
        interaction_enrich_null = {m:pairwise_interaction_enrichment_null(
            input_model = lasso_1.models[m],
            coefficient_order = list(range(2, lasso_1.max_interaction_order+1)),
            seed = seed,
            int_feat_dict = int_feat_dict) for m in lasso_1.models.keys()}
        interaction_enrich_order2only = [lasso_1.pairwise_interaction_enrichment(
                input_model = lasso_1.models[m],
                coefficient_order = [2]) for m in lasso_1.models.keys()]
        interaction_enrich_order3only = [lasso_1.pairwise_interaction_enrichment(
                input_model = lasso_1.models[m],
                coefficient_order = [3]) for m in lasso_1.models.keys()]
        interaction_enrich_order4only = [lasso_1.pairwise_interaction_enrichment(
                input_model = lasso_1.models[m],
                coefficient_order = [4]) for m in lasso_1.models.keys()]
        lasso_1_df = pd.DataFrame({
            'seed': np.array([seed for i in lasso_1.models.keys()]),
            'type': np.array(['1hot' for i in lasso_1.models.keys()]),
            'test_size': np.array(list(lasso_1.models.keys())),
            'train_size_perc': np.array([int(round(1.0-float(i), 2)*100) for i in list(lasso_1.models.keys())]),
            'test_corr': np.array([lasso_1.models[m].test_corr for m in lasso_1.models.keys()]),
            'test_pvar': np.array([np.power(lasso_1.models[m].test_corr, 2) for m in lasso_1.models.keys()]),
            'test_pvarexpl': np.array([np.power(lasso_1.models[m].test_corr, 2)/var_expl for m in lasso_1.models.keys()]),
            'n_coef': np.array([lasso_1.models[m].coef_nonzero for m in lasso_1.models.keys()]),
            'n_coef_0': np.array([len([j for j in i if j=="WT"]) for i in coef_nzero]),
            'n_coef_1': np.array([len([j for j in i if (len(j.split('_'))==1) & (j!="WT")]) for i in coef_nzero]),
            'n_coef_2': np.array([len([j for j in i if (len(j.split('_'))==2)]) for i in coef_nzero]),
            'n_coef_3': np.array([len([j for j in i if (len(j.split('_'))==3)]) for i in coef_nzero]),
            'n_coef_4': np.array([len([j for j in i if (len(j.split('_'))==4)]) for i in coef_nzero]),
            'n_coef_5': np.array([len([j for j in i if (len(j.split('_'))>4)]) for i in coef_nzero]),
            'odds_ratio': np.array([lasso_1.models[m].interaction_enrich['odd_ratio'] for m in lasso_1.models.keys()]),
            'odds_ratio_null': np.array([interaction_enrich_null[m]['odd_ratio'] for m in lasso_1.models.keys()]),
            'p_value': np.array([lasso_1.models[m].interaction_enrich['p_value'] for m in lasso_1.models.keys()]),
            'p_value_null': np.array([interaction_enrich_null[m]['p_value'] for m in lasso_1.models.keys()]),
            'nlog10p_value': np.array([-np.log10(lasso_1.models[m].interaction_enrich['p_value']) for m in lasso_1.models.keys()]),
            'nlog10p_value_null': np.array([-np.log10(interaction_enrich_null[m]['p_value']) for m in lasso_1.models.keys()]),
            'odds_ratio_order2only': np.array([i['odd_ratio'] for i in interaction_enrich_order2only]),
            'p_value_order2only': np.array([i['p_value'] for i in interaction_enrich_order2only]),
            'nlog10p_value_order2only': np.array([-np.log10(i['p_value']) for i in interaction_enrich_order2only]),
            'odds_ratio_order3only': np.array([i['odd_ratio'] for i in interaction_enrich_order3only]),
            'p_value_order3only': np.array([i['p_value'] for i in interaction_enrich_order3only]),
            'nlog10p_value_order3only': np.array([-np.log10(i['p_value']) for i in interaction_enrich_order3only]),
            'odds_ratio_order4only': np.array([i['odd_ratio'] for i in interaction_enrich_order4only]),
            'p_value_order4only': np.array([i['p_value'] for i in interaction_enrich_order4only]),
            'nlog10p_value_order4only': np.array([-np.log10(i['p_value']) for i in interaction_enrich_order4only])})
        #Ensemble
        coef_nzero = [[lasso_e.models[m].feature_names_in_[i] for i in range(len(lasso_e.models[m].feature_names_in_)) if lasso_e.models[m].coef_[i]!=0] for m in lasso_e.models.keys()]
        coef_nzero_int = [[j for j in i if len(j.split("_"))>=2] for i in coef_nzero]
        interaction_enrich_null = {m:pairwise_interaction_enrichment_null(
            input_model = lasso_e.models[m],
            coefficient_order = list(range(2, lasso_e.max_interaction_order+1)),
            seed = seed,
            int_feat_dict = int_feat_dict) for m in lasso_e.models.keys()}
        interaction_enrich_order2only = [lasso_e.pairwise_interaction_enrichment(
                input_model = lasso_e.models[m],
                coefficient_order = [2]) for m in lasso_e.models.keys()]
        interaction_enrich_order3only = [lasso_e.pairwise_interaction_enrichment(
                input_model = lasso_e.models[m],
                coefficient_order = [3]) for m in lasso_e.models.keys()]
        interaction_enrich_order4only = [lasso_e.pairwise_interaction_enrichment(
                input_model = lasso_e.models[m],
                coefficient_order = [4]) for m in lasso_e.models.keys()]
        lasso_e_df = pd.DataFrame({
            'seed': np.array([seed for i in lasso_e.models.keys()]),
            'type': np.array(['ensemble' for i in lasso_e.models.keys()]),
            'test_size': np.array(list(lasso_e.models.keys())),
            'train_size_perc': np.array([int(round(1.0-float(i), 2)*100) for i in list(lasso_e.models.keys())]),
            'test_corr': np.array([lasso_e.models[m].test_corr for m in lasso_e.models.keys()]),
            'test_pvar': np.array([np.power(lasso_e.models[m].test_corr, 2) for m in lasso_e.models.keys()]),
            'test_pvarexpl': np.array([np.power(lasso_e.models[m].test_corr, 2)/var_expl for m in lasso_e.models.keys()]),
            'n_coef': np.array([lasso_e.models[m].coef_nonzero for m in lasso_e.models.keys()]),
            'n_coef_0': np.array([len([j for j in i if j=="WT"]) for i in coef_nzero]),
            'n_coef_1': np.array([len([j for j in i if (len(j.split('_'))==1) & (j!="WT")]) for i in coef_nzero]),
            'n_coef_2': np.array([len([j for j in i if (len(j.split('_'))==2)]) for i in coef_nzero]),
            'n_coef_3': np.array([len([j for j in i if (len(j.split('_'))==3)]) for i in coef_nzero]),
            'n_coef_4': np.array([len([j for j in i if (len(j.split('_'))==4)]) for i in coef_nzero]),
            'n_coef_5': np.array([len([j for j in i if (len(j.split('_'))>4)]) for i in coef_nzero]),
            'odds_ratio': np.array([lasso_e.models[m].interaction_enrich['odd_ratio'] for m in lasso_e.models.keys()]),
            'odds_ratio_null': np.array([interaction_enrich_null[m]['odd_ratio'] for m in lasso_e.models.keys()]),
            'p_value': np.array([lasso_e.models[m].interaction_enrich['p_value'] for m in lasso_e.models.keys()]),
            'p_value_null': np.array([interaction_enrich_null[m]['p_value'] for m in lasso_e.models.keys()]),
            'nlog10p_value': np.array([-np.log10(lasso_e.models[m].interaction_enrich['p_value']) for m in lasso_e.models.keys()]),
            'nlog10p_value_null': np.array([-np.log10(interaction_enrich_null[m]['p_value']) for m in lasso_e.models.keys()]),
            'odds_ratio_order2only': np.array([i['odd_ratio'] for i in interaction_enrich_order2only]),
            'p_value_order2only': np.array([i['p_value'] for i in interaction_enrich_order2only]),
            'nlog10p_value_order2only': np.array([-np.log10(i['p_value']) for i in interaction_enrich_order2only]),
            'odds_ratio_order3only': np.array([i['odd_ratio'] for i in interaction_enrich_order3only]),
            'p_value_order3only': np.array([i['p_value'] for i in interaction_enrich_order3only]),
            'nlog10p_value_order3only': np.array([-np.log10(i['p_value']) for i in interaction_enrich_order3only]),
            'odds_ratio_order4only': np.array([i['odd_ratio'] for i in interaction_enrich_order4only]),
            'p_value_order4only': np.array([i['p_value'] for i in interaction_enrich_order4only]),
            'nlog10p_value_order4only': np.array([-np.log10(i['p_value']) for i in interaction_enrich_order4only])})
        df_list += [pd.concat([lasso_1_df, lasso_e_df], axis=0)]
    #Merge
    plot_df = pd.concat(df_list, axis = 0)
    #Remove infinite odds ratios (only the case when test_size = 0.99)
    plot_df = plot_df.loc[(plot_df['odds_ratio']!=np.inf) & (plot_df['odds_ratio_null']!=np.inf),:]

    #Save
    plot_df.to_csv(os.path.join(plot_dir, 'plot_df.txt'), sep = '\t', index = False)

    #######################################################################
    ## PLOTS ##
    #######################################################################

    #Load
    plot_df = pd.read_csv(os.path.join(plot_dir, 'plot_df.txt'), sep = "\t")

    #Save median model sizes
    grouped = plot_df.groupby(["type", "train_size_perc"])
    pd.DataFrame(grouped.apply(lambda x: np.median(x['n_coef']))).reset_index().to_csv(os.path.join(plot_dir, 'size_median.txt'), sep = '\t', index = False)

    #Save mean enrichment P-values
    grouped = plot_df.groupby(["type", "train_size_perc"])
    pd.DataFrame(grouped.apply(lambda x: np.power(10, (-np.mean(x['nlog10p_value']))))).reset_index().to_csv(os.path.join(plot_dir, 'p_value_mean.txt'), sep = '\t', index = False)

    #Plot palette
    plot_palette = sns.color_palette(["blue", "red"])

    #1
    fig = plt.figure(figsize=(4,2))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="odds_ratio", hue="type", palette=plot_palette, estimator=np.nanmean, errorbar=('ci', 95), errwidth=1)
    plot.axhline(1, color = 'black', linestyle = '--', linewidth = 1)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Odds ratio")
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_oddsratio.pdf'), format="pdf", bbox_inches="tight")
    plt.show()
    ylimits = plot.axes.get_ylim()

    #1a
    fig = plt.figure(figsize=(4,2))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="odds_ratio_null", hue="type", palette=plot_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=1)
    plot.axhline(1, color = 'black', linestyle = '--', linewidth = 1)
    plt.ylim(ylimits)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Odds ratio")
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_oddsratio_null.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #2
    fig = plt.figure(figsize=(4,2))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="nlog10p_value", hue="type", palette=plot_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=1)
    plot.axhline(-np.log10(0.05), color = 'black', linestyle = '--', linewidth = 1)
    plt.xlabel("Training set size (%)")
    plt.ylabel("-log10(P-value)")
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_pvalue.pdf'), format="pdf", bbox_inches="tight")
    plt.show()
    ylimits = plot.axes.get_ylim()

    #2a
    fig = plt.figure(figsize=(4,2))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="nlog10p_value_null", hue="type", palette=plot_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=1)
    plot.axhline(-np.log10(0.05), color = 'black', linestyle = '--', linewidth = 1)
    plt.ylim(ylimits)
    plt.xlabel("Training set size (%)")
    plt.ylabel("-log10(P-value)")
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_pvalue_null.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #3
    fig = plt.figure(figsize=(4,2))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="test_pvar", hue="type", palette=plot_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=1)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Test R-squared")
    plt.title("Sparse model performance on test data")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'performance.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #3a
    fig = plt.figure(figsize=(4,2))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="test_pvarexpl", hue="type", palette=plot_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=1)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Perc. explainable variance")
    plt.title("Sparse model performance on test data")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'performance_explainable.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #4
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="n_coef", hue="type", palette=plot_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=1)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Number of terms")
    plt.title("Sparse model complexity")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'size.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #4a
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df, x="train_size_perc", y="n_coef", hue="type", palette=plot_palette, estimator=np.median, errorbar=('ci', 95), errwidth=1)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Number of terms")
    plt.title("Sparse model complexity")
    plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'size_median.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #5a
    my_palette = custom_linear_colour_palette(['black', 'white'], 5, 1/5.0, 4/5.0)
    plot_df_wide = pd.wide_to_long(
        df = plot_df[['seed', 'type', 'train_size_perc', 'n_coef_1', 'n_coef_2', 'n_coef_3', 'n_coef_4', 'n_coef_5']], 
        stubnames = 'n_coef_', 
        i = ['seed', 'type', 'train_size_perc'], 
        j = 'coef_order')
    plot_df_wide = plot_df_wide.reset_index()
    plot_df_wide['coef_order'] = plot_df_wide['coef_order'].astype('int')
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df_wide.loc[plot_df_wide['type']=='1hot',:], x="train_size_perc", y="n_coef_", hue="coef_order", palette = my_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=0.5)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Number of terms")
    plt.title("Sparse model complexity")
    # plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'order_1hot.pdf'), format="pdf", bbox_inches="tight")
    plt.show()
    ylimits = plot.axes.get_ylim()

    #5b
    my_palette = custom_linear_colour_palette(['black', 'white'], 5, 1/5.0, 4/5.0)
    plot_df_wide = pd.wide_to_long(
        df = plot_df[['seed', 'type', 'train_size_perc', 'n_coef_1', 'n_coef_2', 'n_coef_3', 'n_coef_4', 'n_coef_5']], 
        stubnames = 'n_coef_', 
        i = ['seed', 'type', 'train_size_perc'], 
        j = 'coef_order')
    plot_df_wide = plot_df_wide.reset_index()
    plot_df_wide['coef_order'] = plot_df_wide['coef_order'].astype('int')
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df_wide.loc[plot_df_wide['type']=='ensemble',:], x="train_size_perc", y="n_coef_", hue="coef_order", palette = my_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=0.5)
    plt.ylim(ylimits)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Number of terms")
    plt.title("Sparse model complexity")
    # plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'order_ensemble.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #6a
    my_palette = custom_linear_colour_palette(['red', 'white'], 3, 0.0, 2/3.0)
    custom_linear_colour_palette
    plot_df_subset = copy.deepcopy(plot_df[['seed', 'type', 'train_size_perc', 'odds_ratio_order2only', 'odds_ratio_order3only', 'odds_ratio_order4only']])
    plot_df_subset.columns = ['seed', 'type', 'train_size_perc', 'odds_ratio_order2', 'odds_ratio_order3', 'odds_ratio_order4']
    plot_df_wide = pd.wide_to_long(
        df = plot_df_subset[['seed', 'type', 'train_size_perc', 'odds_ratio_order2', 'odds_ratio_order3', 'odds_ratio_order4']], 
        stubnames = 'odds_ratio_order', 
        i = ['seed', 'type', 'train_size_perc'], 
        j = 'coef_order')
    plot_df_wide = plot_df_wide.reset_index()
    plot_df_wide['coef_order'] = plot_df_wide['coef_order'].astype('int')
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df_wide.loc[plot_df_wide['type']=='ensemble',:], x="train_size_perc", y="odds_ratio_order", hue="coef_order", palette = my_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=0.5)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Odds ratio")
    plt.axhline(1, color = 'black', linestyle = '--', linewidth = 1)
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    # plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_oddsratio_order234only_ensemble.pdf'), format="pdf", bbox_inches="tight")
    plt.show()
    ylimits = plot.axes.get_ylim()

    #6b
    my_palette = custom_linear_colour_palette(['blue', 'white'], 3, 0.0, 2/3.0)
    custom_linear_colour_palette
    plot_df_subset = copy.deepcopy(plot_df[['seed', 'type', 'train_size_perc', 'odds_ratio_order2only', 'odds_ratio_order3only', 'odds_ratio_order4only']])
    plot_df_subset.columns = ['seed', 'type', 'train_size_perc', 'odds_ratio_order2', 'odds_ratio_order3', 'odds_ratio_order4']
    plot_df_wide = pd.wide_to_long(
        df = plot_df_subset[['seed', 'type', 'train_size_perc', 'odds_ratio_order2', 'odds_ratio_order3', 'odds_ratio_order4']], 
        stubnames = 'odds_ratio_order', 
        i = ['seed', 'type', 'train_size_perc'], 
        j = 'coef_order')
    plot_df_wide = plot_df_wide.reset_index()
    plot_df_wide['coef_order'] = plot_df_wide['coef_order'].astype('int')
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df_wide.loc[plot_df_wide['type']=='1hot',:], x="train_size_perc", y="odds_ratio_order", hue="coef_order", palette = my_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=0.5)
    plt.ylim(ylimits)
    plt.xlabel("Training set size (%)")
    plt.ylabel("Odds ratio")
    plt.axhline(1, color = 'black', linestyle = '--', linewidth = 1)
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    # plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_oddsratio_order234only_1hot.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

    #6c
    my_palette = custom_linear_colour_palette(['red', 'white'], 3, 0.0, 2/3.0)
    custom_linear_colour_palette
    plot_df_subset = copy.deepcopy(plot_df[['seed', 'type', 'train_size_perc', 'nlog10p_value_order2only', 'nlog10p_value_order3only', 'nlog10p_value_order4only']])
    plot_df_subset.columns = ['seed', 'type', 'train_size_perc', 'nlog10p_value_order2', 'nlog10p_value_order3', 'nlog10p_value_order4']
    plot_df_wide = pd.wide_to_long(
        df = plot_df_subset[['seed', 'type', 'train_size_perc', 'nlog10p_value_order2', 'nlog10p_value_order3', 'nlog10p_value_order4']], 
        stubnames = 'nlog10p_value_order', 
        i = ['seed', 'type', 'train_size_perc'], 
        j = 'coef_order')
    plot_df_wide = plot_df_wide.reset_index()
    plot_df_wide['coef_order'] = plot_df_wide['coef_order'].astype('int')
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df_wide.loc[plot_df_wide['type']=='ensemble',:], x="train_size_perc", y="nlog10p_value_order", hue="coef_order", palette = my_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=0.5)
    plt.xlabel("Training set size (%)")
    plt.ylabel("-log10(P-value)")
    plt.axhline(1, color = 'black', linestyle = '--', linewidth = 1)
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    # plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_pvalue_order234only_ensemble.pdf'), format="pdf", bbox_inches="tight")
    plt.show()
    ylimits = plot.axes.get_ylim()

    #6d
    my_palette = custom_linear_colour_palette(['blue', 'white'], 3, 0.0, 2/3.0)
    custom_linear_colour_palette
    plot_df_subset = copy.deepcopy(plot_df[['seed', 'type', 'train_size_perc', 'nlog10p_value_order2only', 'nlog10p_value_order3only', 'nlog10p_value_order4only']])
    plot_df_subset.columns = ['seed', 'type', 'train_size_perc', 'nlog10p_value_order2', 'nlog10p_value_order3', 'nlog10p_value_order4']
    plot_df_wide = pd.wide_to_long(
        df = plot_df_subset[['seed', 'type', 'train_size_perc', 'nlog10p_value_order2', 'nlog10p_value_order3', 'nlog10p_value_order4']], 
        stubnames = 'nlog10p_value_order', 
        i = ['seed', 'type', 'train_size_perc'], 
        j = 'coef_order')
    plot_df_wide = plot_df_wide.reset_index()
    plot_df_wide['coef_order'] = plot_df_wide['coef_order'].astype('int')
    fig = plt.figure(figsize=(4,4))
    plot = sns.barplot(data=plot_df_wide.loc[plot_df_wide['type']=='1hot',:], x="train_size_perc", y="nlog10p_value_order", hue="coef_order", palette = my_palette, estimator=np.mean, errorbar=('ci', 95), errwidth=0.5)
    plt.ylim(ylimits)
    plt.xlabel("Training set size (%)")
    plt.ylabel("-log10(P-value)")
    plt.axhline(1, color = 'black', linestyle = '--', linewidth = 1)
    plt.title("Sparse model coefficient enrichment for\npairwise physical interactions")
    # plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(plot_dir, 'enrich_pvalue_order234only_1hot.pdf'), format="pdf", bbox_inches="tight")
    plt.show()

#Run
main()



