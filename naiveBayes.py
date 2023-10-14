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
    self.automaticTuning = True # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
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
    #Calculate counts and probabilities = P(A|B) = P(B|A) * P(A) / P(B), 
    #P(A) = Prior_Probability or marginal probability, P(A|B) = Posterior Probability, P(B|A) = Likelihood, P(B) Evidence
    prior_probability, conditional_prob, counts = self.computeCounts(trainingData, trainingLabels)

    best_params = self.findBestSmoothingParameter(prior_probability, conditional_prob, counts, kgrid, validationData, validationLabels)

    # Set the best parameters after tuning
    self.prior, self.conditionalProb, self.k = best_params

  def computeCounts(self, trainingData, trainingLabels):
    """
    Compute prior, conditional probabilities, and counts from training data.
    """
    prior_probability = util.Counter()
    conditional_prob = util.Counter()
    counts = util.Counter()

    for datum, label in zip(trainingData, trainingLabels):
        prior_probability[label] += 1
        for feat, value in datum.items():
            counts[(feat, label)] += 1
            if value > 0:  #Assume binary
                conditional_prob[(feat, label)] += 1

    return prior_probability, conditional_prob, counts    
  def findBestSmoothingParameter(self, common_prior, common_conditional_prob, common_counts, kgrid, validationData, validationLabels):
    """
    Find the best smoothing parameter k based on accuracy on the validation set.
    """
    best_accuracy = -1
    best_params = None
    
    for k in kgrid:
        prior, conditional_prob, counts = self.applySmoothing(k, common_prior, common_conditional_prob, common_counts)

        accuracy = self.evaluateAccuracy(validationData, validationLabels, prior, conditional_prob)

        if accuracy > best_accuracy:
            best_params = (prior, conditional_prob, k)
            best_accuracy = accuracy

        print(f"Performance validation set for k={k:.3f}: {accuracy * 100:.1f}%")
        
    return best_params
    
  def applySmoothing(self, k, common_prior, common_conditional_prob, common_counts):
    """
    Apply Laplace smoothing to the common counts and probabilities for a given k.
    """
    prior = common_prior.copy()
    conditional_prob = common_conditional_prob.copy()
    counts = common_counts.copy()

    for label in self.legalLabels:
        for feat in self.features:
            conditional_prob[(feat, label)] += k
            counts[(feat, label)] += 2 * k  # 2 because both value 0 and 1 are smoothed

    prior.normalize()
    for x, count in conditional_prob.items():
        conditional_prob[x] = count / counts[x]

    return prior, conditional_prob, counts
    
  def evaluateAccuracy(self, validationData, validationLabels, prior, conditional_prob):
    """
    Evaluate the accuracy of the classifier on the validation set using the given parameters.
    """
    predictions = self.classify(validationData)
    accuracy = sum(predictions[i] == validationLabels[i] for i in range(len(validationLabels))) / len(validationLabels)
    return accuracy
    
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
        #Formula for odds ratio: P(feature=1 | label1) / P(feature=1 | label2)
        odds_ratio = (self.conditionalProb[(feat, label1)] / self.conditionalProb[(feat, label2)])
        featuresOdds.append((odds_ratio, feat))

    # Sort and take the top 100
    featuresOdds.sort(reverse=True)
    featuresOdds = [feat for _, feat in featuresOdds[:100]]    
    #util.raiseNotDefined()

    return featuresOdds
