

# ai_fairness
Stories of machine learning algorithms are in the news; hiring algorithms with male bias, recidivism risk predictors biasing against people of color,image detectors not recognizing darker skinned faces. Increasingly companies, governments,and individuals are moving to machine learning algorithms to lower the decision making burden.   

If the data used to make the decisions has bias,the machine learning algorithm will perpetuate this bias under the guise of impartiality. This provides cover for the human decision maker by passing the blame on the computer. Companies and governments who use these algorithms face a huge liability if their algorithms can be proved to be discriminatory to protected classes. 






### AI Fairness 360 Toolkit 
This project explores and applies metrics and algorithms from the AI Fairness 360
toolkit developed by IBM Trusted AI to mitigate bias in machine learning.  It is specificall designed to work with binary classification.   The user also specifies which features in the data are considered protected attributes and within the protected attributes which groups are privileged or unprivileged.  

![Code snippet](https://github.com/branlindsey/ai_fairness/blob/master/images/Screen%20Shot%202020-06-09%20at%209.16.12%20PM.png)

-Fairness Metrics
- Mean Difference - The mean difference is the difference in the rate of receiving the benefit by privileged or unprivileged group.  
- Disparate Impact - The disparate impact is the ratio who receives the benefit in the privileged and unprivileged groups.  We would like these to be as close to zero as possible.  I used 1 - Disparate Impact because it allowed both metrics to converge to 0 as bias was mitigated.  

- Fairness Algorithms Used 

- Preprocessing 
-- Reweighing
- Inprocessing
-- Adversarial Debiasing
- Postprocessing
-- Calibrated Equal Odds ?


## Does bias exist in the data from Company X? 

At around \$100,000, there is clear drop of the amount of women at the company while the quantity of men stays relatively fixed.   
![salary_image](https://github.com/branlindsey/ai_fairness/blob/master/images/salaries_edited.png)

## Will machine learning amplify the bias that already exists in the data? 
One of the major concerns with bias in machine learning is that an algorithms can actaully amplify the bias already present in the data. 
In order to choose a salary threshold, I looked at the 

![Dataset Metrics at Different Salary Thresholds]

![Classifier Metrics at Different Salary Thresholds]

#### Initial Random Forest Classifier with Gender Removed 

## What algorithms 



![model_metrics](https://github.com/branlindsey/ai_fairness/blob/master/images/model_150_metrics.png)
 I reduced bias by 33% in the final Reweighed Random Forest model while only losing 1% of accuracy to the original model. 
### Resources:
https://aif360.mybluemix.net/
http://www.datasciencepublicpolicy.org/projects/aequitas/
https://fairlearn.github.io/

### Further Reading 
https://hbr.org/2016/12/a-guide-to-solving-social-problems-with-machine-learning
https://blog.insightdatascience.com/tackling-discrimination-in-machine-learning-5c95fde95e95
https://towardsdatascience.com/artificial-intelligence-fairness-and-tradeoffs-ce11ac284b63
