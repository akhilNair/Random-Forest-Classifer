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

y = []
X_drug = []
X_drug_test = []
rule_list = []
rule_list_test = []
y_final_test = []

with open('akhil_train_file.csv','r') as file:
    data_list = csv.reader(file,delimiter='|')
    for row in data_list:
        X_drug.append(row[0])
        data = re.sub('~',' ',row[1])
        rule_list.append(data)
        y.append(row[2])

#tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)
tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_df= 0.85, min_df= 0.01)

#training data for drug
X_drug_tfidf = tfidf.fit_transform(X_drug)

#fit_transform() returns a tf-idf-weighted document-term matrix.
tfidf_rule = tfidf.fit_transform(rule_list)

'''
#the tuple represents, document no. (in this case sentence no.) and feature no.
print(tfidf_rule)

#to understand the tfidf matrix
for i, feature in enumerate(tfidf.get_feature_names()):
    print(i, feature)
'''

#* Technically your TFIDF is just a matrix where the rows are records and the columns are features.
# As such to combine you can append your new features as columns to the end of the matrix.

#tfidf returns sparse matrix. Hence convert to dense matrix
dense_rule_list = array(tfidf_rule.todense())
dense_drug = array(X_drug_tfidf.todense())

row2,column2 = dense_drug.shape
print(dense_rule_list.shape)

# for dense_rule in dense_rule_list:
#     print(dense_rule.shape)
row1, column1 = dense_rule_list.shape
col = max(column1,column2)
dense_rule_list.resize(row1, col)

dense_drug.resize(row1,col)


final_dense =dense_rule_list + dense_drug

X = sparse.csr_matrix(final_dense)


#X_drug_tfidf_train, X_drug_tfidf_test, y_drug_train, y_drug_test = train_test_split(X,y, test_size = 0.3)
#X_rules_tfidf_train, X_rules_tfidf_test, y_rules_train, y_rules_test = train_test_split(X_rules_tfidf,y, test_size = 0.3)

#train and split on the training file for testing

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

rcf_headline = RandomForestClassifier(n_estimators=100,n_jobs=3)

rcf_headline.fit(X_train, y_train)
y_rc_headline_pred = rcf_headline.predict(X_test)

print("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_test, y_rc_headline_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_test, y_rc_headline_pred)*100) )


with open('akhil_test_file.csv','r') as file:
    data_list = csv.reader(file,delimiter='|')
    for row in data_list:
        X_drug_test.append(row[0])
        data = re.sub('~',' ',row[1])
        rule_list_test.append(data)
        y_final_test.append(row[2])

#testing data for drug
X_drug_tfidf_test = tfidf.transform(X_drug_test)

#fit_transform() returns a tf-idf-weighted document-term matrix.
tfidf_rule_test = tfidf.transform(rule_list_test)

#tfidf returns sparse matrix. Hence convert to dense matrix
dense_rule_list_test = array(tfidf_rule_test.todense())
dense_drug_test = array(X_drug_tfidf_test.todense())

row2_test,column2_test = dense_drug_test.shape
print(dense_rule_list_test.shape)

# for dense_rule in dense_rule_list:
#     print(dense_rule.shape)
row1_test, column1_test = dense_rule_list_test.shape
col_test = max(column1_test,column2_test)
dense_rule_list.resize(row1_test, col_test)

dense_drug_test.resize(row1_test,col_test)


final_dense_test =dense_rule_list_test + dense_drug_test

X_test2 = sparse.csr_matrix(final_dense_test)


final_prediction = rcf_headline.predict(X_test2)

print("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_final_test, final_prediction, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_final_test, final_prediction)*100) )
