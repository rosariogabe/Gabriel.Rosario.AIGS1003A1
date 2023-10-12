# Gabriel.Rosario.AIGS1003A1
Gabriel Rosario AIGS1003 Assignment 1

An Introduction to Naïve Bayes Classifier
From theory to practice, learn underlying principles of Naïve Bayes

The goal of Naïve Bayes Classifier is to calculate conditional probability:
for each of K possible outcomes or classes Ck.
Let x=(x1,x2,…,xn). Using Bayesian theorem

https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf

Laplace smoothing in Naïve Bayes algorithm
Solving the zero probability problem in Naïve Bayes algorithm

"(venv) PS C:\Users\garosario\OneDrive - Loyalist College\ML_Assignment\Q2Python3> python dataClassifier.py -c naiveBayes --autotune "


M:\ML_Assignment\REPO>dataClassifier.py -c naiveBayes --autotune --training 700
Doing classification
--------------------
data:           digits
classifier:             naiveBayes
training set size:      700
using automatic tuning for naivebayes
Extracting features...
Training...
Performance on validation set for k=0.001: 81.0%
Performance on validation set for k=0.010: 80.3%
Performance on validation set for k=0.050: 80.0%
Performance on validation set for k=0.100: 79.7%
Performance on validation set for k=0.500: 79.0%
Performance on validation set for k=1.000: 79.0%
Performance on validation set for k=5.000: 77.0%
Performance on validation set for k=10.000: 73.3%
Performance on validation set for k=20.000: 65.3%
Performance on validation set for k=50.000: 53.7%
Validating...
243 correct out of 300 (81.0%).
Testing...
211 correct out of 300 (70.3%).
===================================
Mistake on example 2
Predicted 8; truth is 2
Image:




           +++#####+
          +#########+
          +##########+
           ++++  +###+
                  +##+
                 +###+
                +###+
               +###+
              +####+
              +###+
             +###+
            +###+
           +###+
          +###+
         +####+
        +####+  ++++
       +#####++###++
      +##########+
       +#######+
         ++++





M:\ML_Assignment\REPO>


Classify the following statements, “a cup of hot coffee” and “a cone of ice cream”, given the categories Sunny and Rainy. The following is a training data:
Training Data
Expression	Category	Statements
it is raining	rainy	a cup of hot coffee
picnic on a hot afternoon	sunny	a cone of ice cream
they wore sunglasses	sunny	a cone of ice cream
going out with an umbrella	rainy	a cup of hot coffee



Creating word features Assuming that every word is independent of the other ones:
1)
P(a cone of ice cream) = P(a) * P(cone) * P(of) * P(ice) * P(cream)
P(a cone of ice cream|sunny) =  P(a|sunny) * P(cone|sunny) * P(of|sunny) * P(ice|sunny) * P(cream|sunny)

2)
P(a cup of hot coffee) =  P(a) * P(cup) * P(of) * P(hot) * P(coffee)
P(rainy|a cup of hot coffee) =  P(a|rainy) * P(cup|rainy) * P(of|rainy) * P(hot|rainy) * P(coffee|rainy)

For each category ("Sunny" or "Rainy")
As we see we, maximizes the posterior probability based on Bayes' Theorem
With the Conditional independence assumption, we can express that way.
So, for each statement and category, we will decompose the joint distribution into a product of conditional probabilities for each word in the statement based on the category C. The category with the higher posterior probability will be the predicted category for the statement.
Now we have every word in our training dataset several times for calculate the probabilities. 
•	Calculating Probabilities: Calculate the probability is counting in our training data.
•	Priori Probability of each tag P(sunny).
•	Normalize to make them sum 1.
•	Correct the Zero Probability: Apply Laplace smoothing adding 1 to every count so it’s never zero.
Calculating then P(A|B) : by counting how many times the word of our new sets appears and divide by the total number of the words in Sunny.

https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

