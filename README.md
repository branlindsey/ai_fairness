

# ai_fairness
Stories of machine learning algorithms are in the news; hiring algorithms with male bias, recidivism risk predictors biasing against people of color,image detectors not recognizing darker skinned faces. Increasingly companies, governments,and individuals are moving to machine learning algorithms to lower the decision making burden.   

If the data used to make the decisions has bias,the machine learning algorithm will perpetuate this bias under the guise of impartiality. This provides cover for the human decision maker by passing the blame on the computer. Companies and governments who use these algorithms face a huge liability if their algorithms can be proved to be discriminatory to protected classes. 


Terms
- Bias Amplification



Fairness Metrics
- Mean Difference - The mean difference is the difference in the rate of receiving the benefit by privileged or unprivileged group.  
- Disparate Impact - The disparate impact is the ratio who receives the benefit in the privileged and unprivileged groups.  We would like these to be as close to zero as possible.  I used 1 - Disparate Impact because it allowed both metrics to converge to 0 as bias was mitigated.  

### AI Fairness 360 Toolkit 
This project explores and applies metrics and algorithms from the AI Fairness 360
toolkit developed by IBM Trusted AI to mitigate bias in machine learning. 


Statistical Parity Difference (Mean Difference)
- Disparate Impact 
- 
Fairness Algorithms Used 
- Preprocessing
-- Reweighing
- Inprocessing
-- Adversarial Debiasing
- Postprocessing
-- Calibrated Equal Odds ?


## Does bias exist in the data from Company X? 


## Will machine learning amplify the bias that already exists in the data? 


## What algorithms 


Resources:
https://aif360.mybluemix.net/
http://www.datasciencepublicpolicy.org/projects/aequitas/
https://fairlearn.github.io/

Further Reading 
https://hbr.org/2016/12/a-guide-to-solving-social-problems-with-machine-learning
https://blog.insightdatascience.com/tackling-discrimination-in-machine-learning-5c95fde95e95
https://towardsdatascience.com/artificial-intelligence-fairness-and-tradeoffs-ce11ac284b63
