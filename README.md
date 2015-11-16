The following commands will help in topic modeling of index files using LDA:

gen_titles.py:
    Usage: python2 gen_titles.py <index_files_directory>
             
    Generates the titles, document frequency matrix, the vector of words and the document names for further usage in topic modeling. It uses all the index files in the input directory to generate these dumps. The index files should be in the custom schema shared with the submission of the assignment.

topicModeling.py:
    Usage: python2 topicModeling.py <docFrequency file> <wordVector file> <docNames file> <titles file>

    Topic modeling using lda library of python. This script takes the document frequency matrix as input and returns the set of possible titles for each document.
