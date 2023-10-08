import util
import classificationMethod
import math

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
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    """
    GABRIEL ROSARIO
    Trains the classifier using Laplace smoothing and selects the best value of k
    based on accuracy on the validation set.

    Args:
    trainingData - list of Counters - Training data.
    trainingLabels - list of Labels for the training data.
    validationData - list of Counters - Validation data.
    validationLabels - list Labels for the validation data.
    kgrid list of float  List of candidate values for the smoothing parameter k.
    """    

    best_accuracy = -1  # Best accuracy on validation set
    best_params = None  # Best parameters (prior, conditionalProb, k)

    # Common training - get all counts from training data
    common_prior = util.Counter()  # Probability over labels
    common_conditional_prob = util.Counter()  # Conditional probability of feature feat being 1
    common_counts = util.Counter()  # How many times I have seen feature feat with label y

    for datum, label in zip(trainingData, trainingLabels):
        common_prior[label] += 1
        for feat, value in datum.items():
            common_counts[(feat, label)] += 1
            if value > 0:  # Assume binary value
                common_conditional_prob[(feat, label)] += 1

    for k in kgrid:  # Smoothing parameter tuning loop
        prior = common_prior.copy()
        conditional_prob = common_conditional_prob.copy()
        counts = common_counts.copy()

        # Apply Laplace smoothing
        for label in self.legalLabels:
            for feat in self.features:
                conditional_prob[(feat, label)] += k
                counts[(feat, label)] += 2 * k  # 2 because both value 0 and 1 are smoothed

        # Normalize probabilities
        prior.normalize()
        for x, count in conditional_prob.items():
            conditional_prob[x] = count / counts[x]

        self.prior = prior
        self.conditionalProb = conditional_prob

        # Evaluate performance on the validation set
        predictions = self.classify(validationData)
        accuracy = sum(predictions[i] == validationLabels[i] for i in range(len(validationLabels))) / len(validationLabels)

        print(f"Performance on validation set for k={k:.3f}: {accuracy * 100:.1f}%")
        if accuracy > best_accuracy:
            best_params = (prior, conditional_prob, k)
            best_accuracy = accuracy

        # Set the best parameters after tuning
        self.prior, self.conditionalProb, self.k = best_params
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
    
    "*** YOUR CODE HERE ***"
        # Formula: log(P(y | x)) = log(P(y)) + Σ [x_i * log(P(x_i | y)) + (1 - x_i) * log(1 - P(x_i | y))]
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
        for feat in self.features
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
