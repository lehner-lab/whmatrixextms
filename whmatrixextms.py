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
        enrichment_interactions = [["1","10"], ["2","9"], ["3","7"]],
        ensemble = False):
        """
        Initialize a DMSData object.

        :param vtable: Pandas DataFrame with DMS data (required).
        :param max_interaction_order: Maximum interaction order (default:1).
        :param enrichment_interactions: Pairwise interactions to test for enrichment in non-zero coefficients (default:[["1","10"], ["2","9"], ["3","7"]]).
        :param ensemble: Ensemble encode features. (default:False).
        :returns: MochiData object.
        """
        #Save attributes
        self.vtable = vtable
        self.max_interaction_order = max_interaction_order
        self.enrichment_interactions = enrichment_interactions
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
        #Nucleotide or peptide sequence?
        if 'aa_seq' in self.vtable.columns:
            self.variantCol = 'aa_seq'
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
        z_feat = [re.sub("[acgtGAVLMIFYWKRHDESTCNQP]", "", all_feat[i]).split("_") for i in range(len(input_model.coef_)) if input_model.coef_[i]==0]
        z_feat = [i for i in z_feat if len(i) in coefficient_order]
        nz_feat = [re.sub("[acgtGAVLMIFYWKRHDESTCNQP]", "", all_feat[i]).split("_") for i in range(len(input_model.coef_)) if input_model.coef_[i]!=0]
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
            interactions = self.enrichment_interactions,
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
    parser.add_argument('--dataset', type = str, help = "dataset name: 'tRNA', 'eqFP611' or 'simulated'")
    parser.add_argument('--max_interaction_order', type = int, default = 8, help = "maximum interaction order (default: 8)")
    parser.add_argument('--seed', type = int, default = 1, help = "random seed for training target data resampling (default: 1)")
    return parser

#######################################################################
## MAIN ##
#######################################################################

def main():
    """
    Main function.

    :returns: Nothing.
    """

    #Globals
    test_sizes = [0.36,0.68,0.84,0.92,0.96,0.98,0.99]
    enrichment_interactions = []
    l1_lambdas = arange(0.005, 0.25, 0.005)

    #Get command line arguments
    parser = init_argparse()
    args = parser.parse_args()

    #Maximum interaction order
    max_interaction_order = args.max_interaction_order

    #Make output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #Load dimsum data
    fitness_df = pd.read_csv(args.fitness_file, sep = None, engine='python', na_values = [''], keep_default_na = False)

    #tRNA dataset settings
    if args.dataset == 'trna':
        #Fitness defined for all biological replicates and at least 10 read counts in all input samples
        fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness1_uncorr'])==False) & (fitness_df['count_e1_s0']>=10),:]
        fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness2_uncorr'])==False) & (fitness_df['count_e2_s0']>=10),:]
        fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness3_uncorr'])==False) & (fitness_df['count_e3_s0']>=10),:]
        fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness4_uncorr'])==False) & (fitness_df['count_e4_s0']>=10),:]
        fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness5_uncorr'])==False) & (fitness_df['count_e5_s0']>=10),:]
        fitness_df = fitness_df.loc[(np.isnan(fitness_df['fitness6_uncorr'])==False) & (fitness_df['count_e6_s0']>=10),:]
        #Pairwise interactions to test for enrichment
        enrichment_interactions = [["1","10"], ["2","9"], ["3","7"]]

    #eqFP611 dataset options
    if args.dataset == 'eqFP611':
        #Pairwise interactions to test for enrichment
        enrichment_interactions = [["3","5"], ["3","9"], ["3","10"], ["5","9"], ["5","10"], ["9","10"]] 

    #Simulated dataset options
    if args.dataset == 'simulated':
        #Pairwise interactions to test for enrichment
        enrichment_interactions = [["1", "2"], ["3", "5"], ["5", "6"]] 

    #######################################################################
    ## LASSO ##
    #######################################################################

    for mtype in ['1hot', 'ensemble']:
        #Construct feature matrix
        lasso_model = SparseModel(        
            vtable = fitness_df,
            max_interaction_order = max_interaction_order,
            enrichment_interactions = enrichment_interactions,
            ensemble = {'1hot': False, 'ensemble': True}[mtype])

        #Fit models
        for test_size in test_sizes:
            print(test_size)
            lasso_model.fit_lasso(
                name = str(test_size),
                l1_lambdas = l1_lambdas,
                test_size = test_size,
                random_state = args.seed)

        #Save
        with open(os.path.join(args.output_dir, 'lasso_'+mtype+"_"+str(args.seed)+'.pkl'), 'wb') as outp:
            pickle.dump(lasso_model, outp, pickle.HIGHEST_PROTOCOL)

#Run
main()


