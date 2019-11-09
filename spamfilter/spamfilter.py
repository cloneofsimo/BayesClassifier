
import numpy as np # linear algebra

from sklearn import svm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input/"))
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time
import zipfile
from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


TRAIN_DIR = r"train-mails"
TEST_DIR = r"test-mails"

#  Function to create a dictionary where non-alphabetical characters or single charcaters are removed
def make_Dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for mail in emails:
#         Extracting each mail datas
        with open(mail, encoding='latin-1') as m:
            for line in m:
                words = line.split()
                all_words += words
#     creating a dictionary of words alog with number of occurences
    dictionary = Counter(all_words)
    list_to_be_removed = dictionary.keys()
    list_to_be_removed = list(list_to_be_removed)
    for item in list_to_be_removed:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
#     Extracting most common 3000 items from the dictionary
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000)) #Creating a Matrix of documents ID vs Word ID
    train_labels = np.zeros(len(files))
    count = 0;
    docID = 0;
    for fil in files:
        with open(fil, encoding='latin-1') as fi:
            for i,line in enumerate(fi):
                if i == 2: # as the Main Text starts in the 3rd line where 1st and 2nd line corresponds to Subject of the mail and a newline respectively
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID,wordID] = words.count(word)

        train_labels[docID] = 0;
        filepathTokens = fil.split('/')
        lastToken = filepathTokens[len(filepathTokens) - 1]
        if lastToken.startswith("spmsg"): # Checks if the file name has "spmsg" in it
            train_labels[docID] = 1; # Marks the label as 1 if the mail name has "spmsg"
            count = count + 1
        print(str(docID) + "completed")
        docID = docID + 1
    return features_matrix, train_labels


class NaiveBayes:
    def __init__(self, trainObv, trainLabs, labelcount):
        self.trainObv = trainObv
        self.trainLabs = trainLabs
        self.labelcount = labelcount
        #Probclass:
        self.probClass = ... # class o 의 빈도수?
        self.likelyhoodMatrix = self.savelikelyhood() #the i,j element of this matrix is likelyhood of feature j in class i, in log scale. i.e., log(P(i)P(j|i))




    def likelyhood(self, feature, class):
        #class 에 포함될 확률은?
        Q = self.probClass[class]
        #trainLab 에서 class 인 녀석들 중, feature 가 있는 녀석은 몇마리인가?  (확률은?)
        P =
        return ln(P*Q)
    def savelikelyhood(self):
        X = np.zeros(self.labelcount, self.trainObv.shape[1])
        for i in range(self.trainObv.shape[1]):
            for j in range(self.labelcount):
                X[j,i] = self.likelyhood(i,j)
        return X

https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/

    def naiveProbLabel(self, testObv):
        testLabs = np.zeros(testObv.shape[0])
        for i in range(testObv.shape[0]):
            P = np.zeros(1,self.labelcount)
            for j in nonemptyIndexof(testObv[i]):
                P += self.likelyhoodMatrix[:j].T
            m = np.argmax(P)
            testLabs[i] = m
        return testLabs

class GaussianNaiveBayes:
    just like above, but with lik



start = time.time() # To check the start time
dictionary = make_Dictionary(TRAIN_DIR) # create a most common Word dictionary along with their counts for 3000 words
print(dictionary)
print ("reading and Extracting emails from file.")

features_matrix = np.load("features_matrix.npy")
labels = np.load("labels.npy")
test_feature_matrix = np.load("test_feature_matrix.npy")
test_labels = np.load("test_labels.npy")


end= time.time() # To check end time
print((end-start)/60) # to check total time taken by the procedure

print(feature_matrix)
