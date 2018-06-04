# Text Classifier
Simple Text Classifier in Python, using Scikit Learn CountVectorizer and Naive Bayes MultinominalNB algorithm.

Datasets dir has three others dirs: 'cartas', 'receitas' and 'testes'.
Directories 'cartas' and 'receitas' was used as training sets, and 'testes' as a test set. 

This exemple have 2 types of texts: Love letters and Cake recipes, both in brazilian portuguese.
For testing, 2 files of each text type inside of 'testes' directory.

The file 'stopWords.txt' has portuguese stop words, which we don't consider for use in countVectorizer function.
