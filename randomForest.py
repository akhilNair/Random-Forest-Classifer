'''
* PCA - dimension reduction technique
* Random forest builts multiple decision tress. Each tree outputs a label
* Its an ensemble algorithm
* It creates a set of decision trees from randomly selected subset of training data
'''

# The TF-IDF vectoriser produces sparse outputs as a scipy CSR matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from scipy import sparse
from numpy import array
import re
import pickle

drug_train = []
drug_test = []
rule_train = []
rule_test = []
y_train = []
y_test = []

with open('akhil_train_file.csv','r') as file:
    data_list = csv.reader(file,delimiter='|')
    for row in data_list:
        drug_train.append(row[0])
        data = re.sub('~',' ',row[1])
        rule_train.append(data)
        y_train.append(row[2])


with open('akhil_test_file.csv','r') as file:
    data_list = csv.reader(file,delimiter='|')
    for row in data_list:
        drug_test.append(row[0])
        data = re.sub('~',' ',row[1])
        rule_test.append(data)
        y_test.append(row[2])

'''
######### Function to concat multiple textual features into a singular one using tfidf   ############
'''

def concat_features(tfidf,drug_train,rule_train,flag):
    if flag == 0:
        # training data for drug
        tfidf_drug = tfidf.fit_transform(drug_train)
        # fit_transform() returns a tf-idf-weighted document-term matrix.
        tfidf_rule = tfidf.fit_transform(rule_train)
    else:
        # training data for drug
        tfidf_drug = tfidf.transform(drug_train)
        # fit_transform() returns a tf-idf-weighted document-term matrix.
        tfidf_rule = tfidf.transform(rule_train)
    '''
    #the tuple represents, document no. (in this case sentence no.) and feature no.
    print(tfidf_rule)
    #to understand the tfidf matrix
    for i, feature in enumerate(tfidf.get_feature_names()):
        print(i, feature)
    '''

    # * Technically your TFIDF is just a matrix where the rows are records and the columns are features.
    # As such to combine you can append your new features as columns to the end of the matrix.
    # tfidf returns sparse matrix. Hence convert to dense matrix
    dense_rule = array(tfidf_rule.todense())
    dense_drug = array(tfidf_drug.todense())

    row1, col1 = dense_drug.shape
    row2, col2 = dense_rule.shape

    #compute max column size for final sparse matrix
    col = max(col1, col2)
    row = max(row1,row2)

    #Resize dense matrices of DT matrix of both features to carry out addition
    dense_rule.resize(row, col)
    dense_drug.resize(row, col)

    #add both dense matrices to create a singular training data
    X_dense = dense_rule + dense_drug

    #Convert the dense matrix(i.e DT matrix) back into sparse matrix
    X = sparse.csr_matrix(X_dense)

    return X

'''
######### Function for modelling and training  ############
'''

def modelling_taining():
    # tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)
    tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, max_df=0.85, min_df=0.01)

    X_train = concat_features(tfidf, drug_train, rule_train, 0)

    rfc = RandomForestClassifier(n_estimators=100, n_jobs=3)
    rfc.fit(X_train, y_train)

    X_test = concat_features(tfidf, drug_test, rule_test, 1)
    final_prediction = rfc.predict(X_test)

    print("Random Forest F1 and Accuracy Scores : \n")
    print("F1 score {:.4}%".format(f1_score(y_test, final_prediction, average='macro') * 100))
    print("Accuracy score {:.4}%".format(accuracy_score(y_test, final_prediction) * 100))
       
    # save the model to disk
    filename = 'rfmodel'
    pickle.dump(rfc, open(filename, 'wb'))

    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    #print(result,'result')

if __name__ == '__main__':
    modelling_taining()
