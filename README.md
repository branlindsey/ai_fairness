

# ai_fairness
Stories of machine learning algorithms are in the news; hiring algorithms with male bias, recidivism risk predictors biasing against people of color,image detectors not recognizing darker skinned faces. Increasingly companies, governments,and individuals are moving to machine learning algorithms to lower the decision making burden.   

If the data used to make the decisions has bias,the machine learning algorithm will perpetuate this bias under the guise of impartiality. This provides cover for the human decision maker by passing the blame on the computer. Companies and governments who use these algorithms face a huge liability if their algorithms can be proved to be discriminatory to protected classes. 


### AI Fairness 360 Toolkit 
This project explores and applies metrics and algorithms from the [AI Fairness 360
toolkit](https://aif360.mybluemix.net/#) developed by IBM Trusted AI to mitigate bias in machine learning.  It is specificall designed to work with binary classification.   The user also specifies which features in the data are considered protected attributes and within the protected attributes which groups are privileged or unprivileged.  

<details >
  <summary>
    Setting Up a Dataset
  </summary>
<img src='https://github.com/branlindsey/ai_fairness/blob/master/images/Screen%20Shot%202020-06-09%20at%209.16.12%20PM.png'>
</details>

**Fairness Metrics Used**
- Mean Difference - The mean difference is the difference in the rate of receiving the benefit by privileged or unprivileged group.  
- Disparate Impact - The disparate impact is the ratio who receives the benefit in the privileged and unprivileged groups.  We would like these to be as close to zero as possible.  I used 1 - Disparate Impact because it allowed both metrics to converge to 0 as bias was mitigated.  

- Balanced Accuracy -  The balanced accuracy adjusts accuracy for unbalanced classifiers by taking the average of the rate of true positives and true negatives. 

- Average Odds Difference - The average odds difference is th average of difference in FPR and TPR for unprivileged and privileged groups. 

**Fairness Algorithms Used** 
- Preprocessing 
  -  Reweighing - 
- Inprocessing
  - Adversarial Debiasing 
  - Prejudice Remover 
- Postprocessing
-   Calibrated Equalized Odds 


## Does bias exist in the data from Company X? 
Overall there are 2 men for every women at Company X.   At around \$100,000, there is clear drop of the amount of women at the company while the quantity of men stays relatively fixed.   
![salary_image](https://github.com/branlindsey/ai_fairness/blob/master/images/salaries_edited.png)

With further inspection of the data, the HR department is the only department where women outnumber by 2 to 1, and this department also has lowest salary range in the company.  Additionally, women supervise fewere staffpeople than men, even in the HR department.  So gender is encoded in the data whether gender is explicity named or not in the data.  

[Tableau Dashboard Exploring Salary, Gender and Total Reports](https://public.tableau.com/views/GenderHiddenintheData/Dashboard1?:display_count=y&:origin=viz_share_link)


## Will machine learning amplify the bias that already exists in the data? 
One of the major concerns with bias in machine learning is that an algorithms can actaully amplify the bias already present in the data. 
In order to choose a salary threshold, I looked at the dataset metrics and classifier metrics at different thresholds. 

![Dataset Metrics at Different Salary Thresholds](https://github.com/branlindsey/ai_fairness/blob/master/images/model_metrics.png)

![Classifier Metrics at Different Salary Thresholds](https://github.com/branlindsey/ai_fairness/blob/master/images/salary_thresholds_classifier_metrics.png)

 I also looked at the salary quartiles in the company to get a sense of the salaries across departments. It was around \$150,000 for the 25% of the company.  I decided to use this salary as the threshold for testing the bias removal algoritms.   

#### Initial Random Forest Classifier with Gender Removed 
In order to see whether removing gender would lower the bias in the dataset, I ran a Random Forest classifier and tested results.  The Random Forest with gender removed amplified the the bias in relation to mean difference. 
|.           |  Mean Difference  | 1- Disparate Impact   |
|-----------:|:-----------|:-----------|
|Training Set|    .144    |   .217    |
|Test Set.   |    .173    |   .256     |
|RF Predictions|   .236    |  .256     |



## Fairness Algorithms  
I decided to try a variety of fairness algorithms to mitigate the bias in the dataset.  
**Fairness Algorithms Used** 
- Preprocessing 
  -  Reweighing - 
- Inprocessing
  - Adversarial Debiasing 
  - Prejudice Remover 
- Postprocessing
-   Calibrated Equalized Odds 


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
