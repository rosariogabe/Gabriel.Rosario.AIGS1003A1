import util
import classificationMethod
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = list(trainingData[0].keys()) # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    best_accuracy = -1  # Best accuracy on the validation set
    best_params = None  # Best parameters (prior, conditionalProb, k)

    for k in kgrid:
        # Split the training data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(trainingData, trainingLabels, test_size=0.2, random_state=42)

        # Create a Naive Bayes classifier with Laplace smoothing (alpha = k)
        nb_classifier = MultinomialNB(alpha=k)

        # Train the classifier on the training data
        nb_classifier.fit(X_train, y_train)

        # Make predictions on the validation data
        y_pred = nb_classifier.predict(X_valid)

        # Calculate the accuracy
        accuracy = accuracy_score(y_valid, y_pred)

        print(f"Performance validation set for k={k:.3f}: {accuracy * 100:.1f}%")

        if accuracy > best_accuracy:
            best_params = (nb_classifier, k)
            best_accuracy = accuracy

    # Set the best classifier and k after tuning
    self.classifier, self.k = best_params

    #util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
        
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    #datum Counter of features    
    #log-joint probabilities for each legal label
    logJoint = util.Counter()

    #Formula: log(P(y | x)) = log(P(y)) + Σ [x_i * log(P(x_i | y)) + (1 - x_i) * log(1 - P(x_i | y))]
    for label in self.legalLabels:
        logJoint[label] = math.log(self.prior[label])
        for feat, value in datum.items():
            conditional_prob = self.conditionalProb[(feat, label)]
            logJoint[label] += value * math.log(conditional_prob) + (1 - value) * math.log(1 - conditional_prob)

    #util.raiseNotDefined()
    
    return logJoint
  
    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        for feat in self.features:
           # Formula for odds ratio: P(feature=1 | label1) / P(feature=1 | label2)
            odds_ratio = (
                self.conditionalProb[(feat, label1)] / self.conditionalProb[(feat, label2)]
            )
            featuresOdds.append((odds_ratio, feat))

        # Sort and take the top 100
        featuresOdds.sort(reverse=True)
        featuresOdds = [feat for _, feat in featuresOdds[:100]]    
        #util.raiseNotDefined()

        return featuresOdds
