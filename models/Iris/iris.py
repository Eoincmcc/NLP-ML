# imports and librarie 

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split                    # splitting up data for model training & eval        
from sklearn.model_selection import cross_val_score                     # cross validation score evaluation
from sklearn.model_selection import StratifiedKFold                     # Stratified K-Fold: create test sets with even distributions of classes
from sklearn.metrics import classification_report                       # Build a text report showing main classification metrics 
from sklearn.metrics import confusion_matrix                            # matrix for eval of false pos, true, neg etc 
from sklearn.metrics import accuracy_score
# models used                              
from sklearn.linear_model import LogisticRegression                     
from sklearn.tree import DecisionTreeClassifier                          
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB                           
from sklearn.svm import SVC


def main():
    load_data()

def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

def add_tokenizer_params(parser: argparse.ArgumentParser):
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")


if __name__ == "__main__":
    main()