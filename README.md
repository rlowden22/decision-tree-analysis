# 5008 Final Research Paper
* Name: Rowan Lowden
* Semester: Summer 2025
* Topic: Decision Tree Algorithm: CART 

## Introduction/Overview

In this report, I will explore and analyze a type of supervised learning algorithm used in machine learning for classification and regression tasks, decision trees. They are a popular algorithm that splits data into smaller subsets based on feature values making them highly useful for both classification and regression tasks due to their simplicity, interpretability, and ability to handle both categorical and numerical data.[3] A decision tree creates a hierarchial tree structure with a root node, branches, internal nodes, and leaf nodes. The root node does not have any incoming branches, only outgoing branches that feed into the internal nodes known as decision nodes. The nodes conduct evaulations to form homogenous subsets called leaf nodes or terminal nodes.  [^1] 

![Decision Tree](./images/basic%20Decision%20tree.png)

Decision Trees solve the problem of classification and regression by creating a model that makes decisions based on feature values. For classification problems, they predict categorical labels, such as determining whether a mushroom is edible or poisonous based on its attributes (e.g., color, size, odor). [3] In regression problems, Decision Trees predict continuous values, like estimating house prices based on features such as size and location [4]. The algorithm recursively splits data at each node based on the most informative features, using criteria like Gini Impurity or Information Gain [5]. One of the key advantages of Decision Trees is their interpretability, as the structure of the tree allows us to trace the path of decisions leading to a final prediction. This makes them a valuable tool in a wide range of real-world applications, including medical diagnosis, credit scoring, and customer segmentation [6]. 


Decision Tree machine learning uses divide and conquer stratey by conducting a greedy searcg to identify the optimal splits within the tree data. The splitting continues to repeat in a top down recursive manner until all the data has been labeled under specific labels. [1]

Desicion Trees have preferences for small trees, which is consistent with the priniciple of parsimony in Occam's Razor. "entities should not be multplied beyond nessessity." This implies decision trees should only add complexity if neccessary because the simpliest explanation is often the best. To reduce the complexity, pruning is typically used which is when branches are removed that split features of low importance. A group of decision trees called a random forest is often used to maintain accuracy, but here we will focus on the individual decision tree. [1]

### Algorithm Development 

Decision Trees have a rich history in the field of machine learning, with their roots tracing back to the 1960s.Hunt's Algorithm, developed in the 1960s, was one of the earliest methods for modeling human learning in psychology, particularly in terms of inductive learning. It was designed to replicate how humans make decisions based on prior experiences. The algorithm was first introduced by J. Hunt and colleagues as part of a broader study into cognitive processes and the way humans organize knowledge. Hunt’s work laid the groundwork for many modern decision tree algorithms used in machine learning today. [7] Then in the 1980's, the earliest form of Decision Trees was introduced by Ross Quinlan with the ID3 algorithm in 1986. It used information gain to split the data at each node, making it one of the first algorithms to use recursive binary splitting for classification [3]. Quinlan's work then evolved into the C4.5 algorithm, introduced in 1993, which improved upon ID3 by incorporating pruning to avoid overfitting, handling both continuous and discrete attributes, and using the gain ratio for more accurate splits [5]. In the 1980s, the CART (Classification and Regression Trees) algorithm was developed by Breiman et al., which introduced the use of Gini Impurity for classification and Mean Squared Error (MSE) for regression, becoming one of the most widely used Decision Tree algorithms today [4]. Over time, Decision Trees have become a cornerstone of machine learning, leading to the development of ensemble methods like Random Forests and Gradient Boosting, which combine multiple trees to improve predictive accuracy and robustness [6].

This paper will begin with an in-depth look at the Decision Tree algorithm, exploring its background, core principles, and historical development. Following the algorithm's overview, we will analyze its time and space complexities to understand its efficiency and scalability. We will then proceed with an empirical analysis, where we apply the CART Decision Tree algorithm to a mushroom classification dataset, evaluating its performance and visualizing key results. The paper will also cover the application of Decision Trees in real-world contexts/research, demonstrating their practical value across various fields. Finally, we will detail the implementation process, highlighting the challenges faced during this learning process and discussing the decisions made throughout. The conclusion will summarize key findings and suggest potential improvements or future research directions.

### Entropy and Gini Impurity 

## Analysis of Algorithm/Datastructure
Make sure to include the following:
- Time Complexity
- Space Complexity
- General analysis of the algorithm/datastructure

## Empirical Analysis
- What is the empirical analysis?
- Provide specific examples / data.
- provide graphs/visuals


## Application
- What is the algorithm/datastructure used for?
- Provide specific examples
- Why is it useful / used in that field area?
- Make sure to provide sources for your information. (research papers using discrete tree algo)
- Disucss the scientfic papers found!


## Implementation
- What language did you use?
- What libraries did you use?
- What were the challenges you faced?
- Provide key points of the algorithm/datastructure implementation, discuss the code.
- If you found code in another language, and then implemented in your own language that is fine - but make sure to document that.


## Summary
- Provide a summary of your findings
- What did you learn?

## References 
[^1]: IBM. 2023. What is a Decision Tree? IBM. https://www.ibm.com/think/topics/decision-trees (accessed July 25, 2025).  

[2] GeeksforGeeks. Supervised Machine Learning. GeeksforGeeks. Retrieved from https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/

[3] Quinlan, J.R. 1986. Induction of decision trees. Machine Learning, 1(1), 81-106. Springer. DOI: 10.1007/BF00116251.

[4] Breiman, L., Friedman, J.H., Olshen, R.A., and Stone, C.J. 1986. Classification and regression trees. Wadsworth & Brooks/Cole. ISBN 0-534-98099-8.

[5] Quinlan, J.R. 1993. C4.5: Programs for machine learning. Morgan Kaufmann Publishers. ISBN: 1-55860-238-0.

[6] Breiman, L. 2001. Random forests. Machine Learning, 45(1), 5-32. DOI: 10.1023/A:1010933404324.

[7] Hunt, E.B., Marin, J., & Stone, P.J. 1966. Experiments in Inductive Inference. Psychological Review, 73(3), 131-161. https://sso.apa.org/apasso/idm/login?CheckAccess=1&UID=1966-12300-001&ERIGHTS_TARGET=https%3A%2F%2Fpsycnet.apa.org%2FdoiLanding%3Fdoi%3D10.1037%252Fh0023706
