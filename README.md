# 5008 Final Research Paper
* Name: Rowan Lowden
* Semester: Summer 2025
* Topic: Decision Tree Algorithm

## Introduction/Overview

In this report, I will explore and analyze a type of supervised learning algorithm used in machine learning for classification and regression tasks, decision trees. They are a popular algorithm that splits data into smaller subsets based on feature values making them highly useful for both classification and regression tasks due to their simplicity, interpretability, and ability to handle both categorical and numerical data.[^3] A decision tree creates a hierarchial tree structure with a root node, branches, internal nodes, and leaf nodes. The root node does not have any incoming branches, only outgoing branches that feed into the internal nodes known as decision nodes. The nodes conduct evaulations to form homogenous subsets called leaf nodes or terminal nodes.  [^1] 

![Decision Tree](./images/basic%20Decision%20tree.png)

Decision Trees solve the problem of classification and regression by creating a model that makes decisions based on feature values. For classification problems, they predict categorical labels, such as determining whether a mushroom is edible or poisonous based on its attributes (e.g., color, size, odor). [^3] In regression problems, Decision Trees predict continuous values, like estimating house prices based on features such as size and location [^4]. The algorithm recursively splits data at each node based on the most informative features, using criteria like Gini Impurity or Information Gain [^5]. One of the key advantages of Decision Trees is their interpretability, as the structure of the tree allows us to trace the path of decisions leading to a final prediction. This makes them a valuable tool in a wide range of real-world applications, including medical diagnosis, credit scoring, and customer segmentation [^6]. 

In this report. I will provide both theoretical and empirical analysis. First I will provide time and space complexities to understand its efficiency and scalability. Then I will proceed with an empirical analysis, where we apply the CART Decision Tree algorithm to a mushroom classification dataset, evaluating its performance and visualizing key results. The paper will also cover the application of Decision Trees in real-world contexts, such as a discussion on the algorithms use in modern scienfitic research demonstrating the practical value across various fields. Lastly, we will detail the implementation process, highlighting the challenges faced during this learning process and discussing the decisions made throughout. The conclusion will summarize key findings and suggest potential improvements or future research directions.


Desicion Trees have preferences for small trees, which is consistent with the priniciple of parsimony in Occam's Razor. "entities should not be multplied beyond nessessity." This implies decision trees should only add complexity if neccessary because the simpliest explanation is often the best. To reduce the complexity, pruning is typically used which is when branches are removed that split features of low importance. A group of decision trees called a random forest is often used to maintain accuracy, but here we will focus on the individual decision tree. [^1]

### Algorithm Development 

Decision Trees have a rich history in the field of machine learning, with their roots tracing back to the 1960s.Hunt's Algorithm, developed in the 1960s, was one of the earliest methods for modeling human learning in psychology, particularly in terms of inductive learning. It was designed to replicate how humans make decisions based on prior experiences. The algorithm was first introduced by J. Hunt and colleagues as part of a broader study into cognitive processes and the way humans organize knowledge. Hunt’s work laid the groundwork for many modern decision tree algorithms used in machine learning today. [^7] Then in the 1980's, the earliest form of Decision Trees was introduced by Ross Quinlan with the ID3 algorithm in 1986. It used information gain to split the data at each node, making it one of the first algorithms to use recursive binary splitting for classification [^3]. Quinlan's work then evolved into the C4.5 algorithm, introduced in 1993, which improved upon ID3 by incorporating pruning to avoid overfitting, handling both continuous and discrete attributes, and using the gain ratio for more accurate splits [^5]. In the 1980s, the CART (Classification and Regression Trees) algorithm was developed by Breiman et al., which introduced the use of Gini Impurity for classification and Mean Squared Error (MSE) for regression, becoming one of the most widely used Decision Tree algorithms today [^4]. Over time, Decision Trees have become a cornerstone of machine learning, leading to the development of ensemble methods like Random Forests and Gradient Boosting, which combine multiple trees to improve predictive accuracy and robustness [^6].


## Theoretical Analysis

### Time and Complexity: Big O

Understanding the time and space complexity of Decision Tree algorithms is crucial for evaluating their efficiency, particularly as the dataset size increases. The time complexity of Decision Trees is influenced by the number of samples and features in the dataset, with the algorithm needing to evaluate all features for each potential split at each node. They use divide and conquer strategy by conducting a greedy search to identify the optimal splits within the tree data. The splitting continues to repeat in a top down recursive manner until all the data has been labeled under specific labels. [^1] 

In the below pseudocode, The outer `while` loop runs recursively, with the depth of the tree increasing as the number of splits grows. At each level, the algorithm evaluates potential splits for each node. The inner `for` loop iterates over all nodes at the current depth, performing a split if the node is valid `(Node(i) > -1)`. For each node, the algorithm evaluates all `m` features to find the best split, typically involving sorting or scanning the data. Since the tree depth is logarithmic in relation to the number of samples, the overall time complexity is $O(n * m * log n)$ where `n` is the number of samples, `m` is the number of features, and $logn$ corresponds to the tree’s maximum depth [^4]. Therefore, the time complexity for building a decision tree is $O(n * m * log n)$, where $n$ is the number of data samples and $m$ is the number of features. This is because of greedy search for the best feature to split the data, involving sorting or scanning through the dataset. The recursive tree-building process continues until the tree is fully grown, which can be computationally expensive for large datasets [^4].

The space complexity of Decision Trees is typically $O(n)$, as the tree structure needs to store both the data at each node and the tree itself. Each node in the tree may contain information about the data split and the distribution of the labels. Additionally, the tree's depth affects both time and space complexity, with deeper trees requiring more memory to store additional nodes. Overfitting, a common issue with decision trees, can also lead to higher space complexity due to the tree's depth increasing unnecessarily [^3].  

### Pseudocode: CART Algorithm 

Understanding the time complexity of the Decision Tree algorithm involves examining how the algorithm builds the tree and evaluates splits at each node. The outer while loop runs for O(log n) iterations, as the depth of the tree grows logarithmically with respect to the number of samples. The inner for loop processes all nodes at each depth level, with the number of nodes doubling at each level (approximately 2^d nodes at depth d). For each node, the algorithm evaluates m features to find the best split, and this evaluation typically involves sorting or scanning through the dataset, which has a time complexity of O(n log n) per feature. Thus, the overall time complexity is O(n * m * log n), where n is the number of samples, m is the number of features, and log n represents the tree depth.

In terms of space complexity, the Decision Tree algorithm requires space to store both the dataset and the tree structure. Each node in the tree stores information about the feature used for splitting, the split value, and possibly the dataset passing through the node. In the worst case, a fully grown tree with n samples has up to 2n - 1 nodes, which results in a space complexity of O(n). This space is primarily consumed by the dataset and the tree structure, which grows linearly with the number of samples (n). Thus, the overall space complexity of Decision Trees is O(n).

Below is the pseudocode for the Decision Tree CART Algorithm [^8]:

```d = 0, endtree = 0

Note(0) = 1, Node(1) = 0, Node(2) = 0

while endtree < 1
    if Node(2d-1) + Node(2d) + .... + Node(2d+1-2) = 2 - 2d+1   
        endtree = 1

    else
        do i = 2d-1, 2d, .... , 2d+1-2
            if Node(i) > -1
                Split tree

            else 
                Node(2i+1) = -1
                Node(2i+2) = -1

            end if

        end do

    end if

d = d + 1

end while
```



### Entropy and Gini Impurity 

### General analysis of the algorithm/datastructure

## Empirical Analysis
- What is the empirical analysis?
- Provide specific examples / data.
- provide graphs/visuals


## Application

Decision trees have become widely used in many scientific fields beyond computer science due to their intuitive structure and ease of interpretation. They are especially valuable in high-stakes fields like healthcare, where transparency and trust in model predictions are critical [^9^11]. Their ability to identify key variables and present rule-based decisions makes them useful for researchers in areas such as medicine, environmental monitoring, and even energy policy [^10]. These models not only provide accurate predictions but also help researchers and decision-makers understand the "why" behind outcomes—supporting insight-driven interventions across diverse disciplines. During my research, I was pleased to find many research papers utilizing decision trees. Below I briefly describe 3 papers I found interesting and how they implmented decision trees.

The first paper by Lu & Ma published in 2020, proposed two different decsision tree models for short term water quality predicitions in the Tualitin River. They aimed to predict short‑term changes in river conditions such as temperature, dissolved oxygen, pH, conductivity, turbidity, and fluorescence. The authors first took the raw enviromental data collected hourly, cleaned it using CEEMDAN, and then trained decision tree based models (Random Forest and Gradient Boosting) to make forecasts on water quality in the near future. Lastly they compared them to conventional baselines or previously collected data to compare the outcomes of the new models. In this research, decision trees were central because they cope well with irregular, multiscale patterns in environmental data, offer robust performance without heavy feature engineering, and provide variable importance to guide monitoring and operations. In practice, the hybrid “CEEMDAN + tree model” design yielded accurate, stable predictions across indicators—useful for proactive management (e.g., anticipating low oxygen events or turbidity spikes) when timely, trustworthy forecasts matter. [^14]

The second research paper I examined, from 2022, is in response to the urgent need for ICU triage tools during the COVID-19 pandemic, Elhazmi et al. conducted a multicenter study across 14 hospitals in Saudi Arabia from March 2020- Oct 2020 to predict 28-day mortality in critically ill COVID-19 patients using decision trees. The researchers trained a C4.5 decision tree model on clinical data from 1,468 patients, incorporating variables such as intubation status, vasopressor use, oxygenation (PaO₂/FiO₂), age, and gender. The decision tree was chosen for its interpretability and bedside usability, offering a transparent, rule-based structure that clinicians could use to quickly assess risk. The final model demonstrated good accuracy (73.1%) with intubation as the root decision node followed by key clinical indicators [^12]. Compared to traditional logistic regression, the tree provided immediate visual reasoning and actionable insights, highlighting the decision tree’s importance as a practical and interpretable tool in real-time critical care settings.

The third research study examined to demonstrate the real world application of the decision tree algorithm was Hajihosseinlou, Asghari, and Shirvani research from 2023. They applied decision tree–based machine learning to support mineral exploration by modeling the likelihood of Mississippi Valley-type (MVT) lead–zinc deposits in Iran’s Varcheh district. The researchers used Light Gradient Boosting Machine (LightGBM), a highly efficient ensemble of decision trees optimized for speed and accuracy, to analyze a wide range of geoscientific data, including geological structures, geochemical indicators, and remote sensing imagery. LightGBM was selected for its ability to handle large, high-dimensional datasets while capturing complex, nonlinear relationships among predictive features. Compared to other models like XGBoost, LightGBM achieved higher precision and recall, identifying 92% of known mineral occurrences within just 10% of the mapped area. The decision tree–based approach was crucial for reducing exploration costs by pinpointing high-prospect zones, making it an invaluable tool for guiding field investigations and drilling programs in economic geology [^13].  

Decision trees are powerful, versatile tools used across many scientific fields because they combine strong predictive performance with clear, interpretable logic. From ICU triage during the COVID-19 pandemic to forecasting river water quality and guiding mineral exploration, decision trees help researchers make sense of complex data and support real-world decisions. Their ability to handle diverse data types, reveal important variables, and produce transparent, rule-based models makes them especially useful in high-stakes fields like healthcare, environmental science, and resource management. As these examples show, decision trees are not just algorithms—they’re practical frameworks for turning data into actionable insight.

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

[^2]: GeeksforGeeks. Supervised Machine Learning. GeeksforGeeks. Retrieved from https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/

[^3]: Quinlan, J.R. 1986. Induction of decision trees. Machine Learning, 1(1), 81-106. Springer. DOI: 10.1007/BF00116251.

[^4]: Breiman, L., Friedman, J.H., Olshen, R.A., and Stone, C.J. 1986. Classification and regression trees. Wadsworth & Brooks/Cole. ISBN 0-534-98099-8.

[^5]: Quinlan, J.R. 1993. C4.5: Programs for machine learning. Morgan Kaufmann Publishers. ISBN: 1-55860-238-0.

[^6]: Breiman, L. 2001. Random forests. Machine Learning, 45(1), 5-32. DOI: 10.1023/A:1010933404324.

[^7]: Hunt, E.B., Marin, J., & Stone, P.J. 1966. Experiments in Inductive Inference. Psychological Review, 73(3), 131-161. https://sso.apa.org/apasso/idm/login?CheckAccess=1&UID=1966-12300-001&ERIGHTS_TARGET=https%3A%2F%2Fpsycnet.apa.org%2FdoiLanding%3Fdoi%3D10.1037%252Fh0023706

[^8]: GeeksforGeeks. 2023. CART (Classification and Regression Tree) in Machine Learning. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/cart-classification-and-regression-tree-in-machine-learning/ (accessed Aug. 4, 2025).

[^9]: Meshram, S. and Naik, D. 2023. Review on Decision Tree Algorithm in Healthcare Applications. International Journal of Advanced Research in Computer and Communication Engineering. https://www.researchgate.net/publication/382748363

[^10]: Alirezaei, M., Asadi, M., and Azizi, M. 2022. Decision Tree Applications in Energy Policy and Planning: A Review. Energies 15, 7 (2022), 2420. DOI: 10.3390/en15072420.

[^11]: Sharma, S. and Singh, A. 2024. Interpretable Machine Learning Techniques: A Systematic Review with a Focus on Decision Trees. Applied Sciences 14, 19 (2024), 8884. DOI: 10.3390/app14198884

[^12]: A. Elhazmi, M. Alshehri, A. Almutairi, A. Alsalemi, G. Almekhlafi, A. Alamri, A. Alharthi, M. Alfayez, et al. 2022. Predicting mortality in critically ill COVID-19 patients using decision tree models: Multicenter cohort study. Journal of Infection and Public Health 15, 6 (2022), 654–661. DOI: https://doi.org/10.1016/j.jiph.2022.03.003

[^13]: S. Hajihosseinlou, O. Asghari, and H. Shirvani. 2023. Application of Light Gradient Boosting Machine (LightGBM) in prospectivity modeling of Mississippi Valley-type Pb–Zn deposits in the Varcheh district, Iran. Natural Resources Research 32, 1 (2023), 487–506. DOI: https://doi.org/10.1007/s11053-022-10007-7

[^14]: H. Lu and X. Ma. 2020. Hybrid decision tree-based models for short-term water quality prediction. Chemosphere 248 (2020), 125988. DOI: https://doi.org/10.1016/j.chemosphere.2020.125988