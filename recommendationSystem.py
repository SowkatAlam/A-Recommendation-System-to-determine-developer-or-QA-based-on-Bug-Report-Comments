from nltk import word_tokenize
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import random
import glob
import codecs
import sys

def readData(dataLocation, variable):
    for filename in glob.glob(dataLocation):
        fin = codecs.open(filename, encoding="utf-8", errors="ignore")
        variable.append(fin.read())
        fin.close()

def extractCommentFeatures(comment, commonWords, commentMap):
    wordLemmatizer = WordNetLemmatizer();
    feature = {}
    
    # Now stem the words of the given comment
    token = [wordLemmatizer.lemmatize(word.lower()) for word in word_tokenize(comment)]

    # Remove common words and add in feature dictionary
    for word in token:
        feature[word]=True
##    for word in token:
##        if word not in commonWords:
##            feature[word]= True
    commentMap[tuple(feature.items())]=comment
    return feature



developer = []
tester =[]

readData('Developer/*', developer);

readData('Tester/*', tester);


mixComments = [(comment, 'Tester') for comment in tester]
mixComments += [(comment, 'Developer') for comment in developer]
random.shuffle(mixComments);

commonWords = stopwords.words("english");

commentMap={}

commentFeatures = [ (extractCommentFeatures(comment, commonWords, commentMap), label) for (comment, label) in mixComments]

print (commentFeatures)

print("------------------------------------")

# Training and testing
print ("...Training and Testing...")

num_folds = 10
size = int(len(commentFeatures) / num_folds)
total_precision = 0.00
total_recall = 0.00
f_measure = 0.00

idx=1

result = []

test_and_train_ratio= int(len(commentFeatures) *.4)

index_test_begin = 0
index_test_end = index_test_begin+ test_and_train_ratio;

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# Split total feature set based on where test_set currently is
train_set = commentFeatures[:index_test_begin]+commentFeatures[index_test_end:]
test_set = commentFeatures[index_test_begin:index_test_end]
classifier = NaiveBayesClassifier.train(train_set)

# Test and record results of classifications
for (comment, real_classification) in test_set:
    
    res =("\n\nBug report comment "+str(idx)+" :      (Showing first 80 characters of the comment)\n")
    res +="-------------------------------------------------------------------------\n"
    res += (commentMap[tuple(comment.items())][0:70] + " ...");
    guess_classification = classifier.classify(comment)
    #res +="\n-------------------------------------------------------------------------\n"
    res +=("\n\nPredicted Role : " + guess_classification)
    res +=("\nActual Role    : " + real_classification)
    idx+=1
    if guess_classification is "Developer":
        if guess_classification is real_classification:
            true_positives += 1
        else:
            false_positives += 1
    elif guess_classification is "Tester":
        if guess_classification is real_classification:
            true_negatives += 1
        else:
            false_negatives += 1
            
    result.append(res)
            
precision=0.00
recall=0.00
fmeasure=0.00

if (true_positives + false_positives) is not 0:
    precision = float(true_positives) / float(true_positives + false_positives)
if (true_positives + false_negatives) is not 0:
    recall = float(true_positives) / float(true_positives + false_negatives)
    
if int(precision + recall) is not 0:
    fmeasure = float ((2 * precision * recall) / (precision + recall))

total_precision += precision
total_recall += recall
f_measure += fmeasure

    
# Report major results for entire training and testing
##total_precision = total_precision / num_folds
##total_recall = total_recall / num_folds
##f_measure = f_measure/ num_folds


for r in result:
    print(r)

print ("\n... Evaluation results ...")

print ("Precision : "+ str(round (total_precision*100,2)) +"%")
print ("Recall : " + str( round(total_recall*100,2)) + "%")
print ("F-measure : " + str( round(f_measure*100,2)) + "%")

print (len(test_set))
print (len(train_set))

##print ("Precision : "+ str(total_precision*100) +"%")
##print ("Recall : " + str( total_recall*100) + "%")
##print ("F Measure : " + str( f_measure*100) + "%")
