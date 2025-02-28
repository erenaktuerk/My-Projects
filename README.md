# My-Projects

Collection and categorization of my projects

## Machine Learning Projects

### [CNN CIFAR-10 Image Classifier](https://github.com/erenaktuerk/cnn-cifar10-classifier)
A robust and highly optimized image classification project that leverages an advanced Convolutional Neural Network (CNN) to accurately classify CIFAR-10 images. This project demonstrates an end-to-end deep learning pipeline that combines state-of-the-art techniques—including hyperparameter tuning, data augmentation, feature map visualization, and Grad-CAM analysis—to create a solution that is both practically applicable and technically impressive.
- **Model Type**: Convolutional Neural Network (CNN)
- **Technologies Used**: Python, TensorFlow, Keras, Keras Tuner, OpenCV, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Key Features**:
  
	•	Advanced CNN Architecture: A multi-block network with several convolutional layers enhanced by Batch Normalization and Dropout, enabling deep feature extraction with minimal overfitting.

	•	Comprehensive Hyperparameter Tuning: Utilizes Keras Tuner to optimize parameters such as filter counts, kernel sizes, dropout rates, dense layer units, and learning rates for near-optimal performance.

	•	Real-Time Data Augmentation: Incorporates dynamic data augmentation techniques to further boost model generalization and robustness in real-world scenarios.

	•	Model Interpretability: Provides in-depth insights with Feature Map Visualizations and Grad-CAM overlays, revealing the inner workings of the CNN and highlighting the critical regions influencing predictions.

	•	Modular End-to-End Pipeline: A seamless integration of data preprocessing, training, evaluation, and visualization that ensures scalability and ease of deployment in production environments.

- **Project Highlights**:

	•	Practical and Production-Ready: Built to solve real-world image classification challenges with a focus on reliability, interpretability, and performance.

	•	Problem-Solving Oriented: Addresses common deep learning issues such as overfitting, hyperparameter selection, and model transparency, making it a comprehensive solution for complex computer vision tasks.

	•	Impressive Performance: Achieves robust accuracy on the CIFAR-10 dataset, validated through extensive evaluation metrics including confusion matrices and classification reports.

	•	Innovative Visual Insights: Advanced visualizations (accuracy/loss curves, feature maps, Grad-CAM) provide a deep understanding of the model’s behavior and decision-making process.

	•	Scalable and Extensible Architecture: Designed with modularity in mind, allowing for future enhancements such as ensemble methods or transfer learning to be incorporated effortlessly.
- **Status**: Completed

### [AAPL Share Prediction](https://github.com/erenaktuerk/aapl-share-prediction)  
A regression model designed to predict Apple Inc. (AAPL) stock prices using historical market data, temporal features (such as time of day, previous closing prices), and advanced machine learning techniques. The project emphasizes data preprocessing, feature engineering, and hyperparameter optimization to achieve high accuracy.

- **Model Type**: Regression
- **Technologies used**: Python, XGBoost, TensorFlow, Pandas, NumPy, Scikit-learn
- **Key features**:

	•	Advanced Data Preprocessing: Thorough cleaning and feature selection to ensure high-quality input data.

	•	Feature Engineering: Derivation of temporal features to capture market trends and stock patterns.

	•	Hyperparameter Optimization: Extensive use of RandomizedSearchCV for model tuning, maximizing predictive accuracy.

	•	Model Comparison: Evaluation of both XGBoost and TensorFlow-based MLP to determine the best-performing model.

	•	Cross-Validation: Comprehensive validation with optimized folds to avoid overfitting and ensure robustness.

- **Project Highlights**:

	•	Problem: Predicting future AAPL stock prices based on historical data to support investment decisions.

	•	Methodology: The project incorporates XGBoost and a TensorFlow-based MLP to assess which model best fits the prediction task. Hyperparameter optimization and data preprocessing techniques are crucial in obtaining high accuracy.

	•	Results & Evaluation: The model performance is evaluated using key metrics like RMSE, MAE, and visualized via relevant plots (e.g., loss curves, performance plots). Cross-validation results further validate model robustness.

	•	Lessons Learned: The project emphasizes the importance of data quality, feature selection, and hyperparameter optimization. Future iterations could focus on expanding the feature set or using more advanced ensemble methods.

- **Status**: Completed

### [BERT Text Classifier](https://github.com/erenaktuerk/bert-text-classifier)  
This project explores the power of BERT (Bidirectional Encoder Representations from Transformers), a cutting-edge NLP model developed by Google Research in 2018. Built on the transformer architecture, BERT’s bidirectional approach enables a deep contextual understanding of language.

In this classification task, BERT is applied to analyze and categorize textual data — specifically user reviews — into predefined sentiment or topic-based classes. The project highlights advanced Natural Language Processing techniques, including thorough data preprocessing, model fine-tuning, and performance evaluation. By leveraging state-of-the-art transformer-based models, the goal is to achieve highly accurate and reliable classification results.

- **Model Type**: Classification
- **Technologies used**: Python, BERT, HuggingFace, PyTorch, Scikit-learn
- **Key features**:

	•	Text Preprocessing: Cleans and tokenizes raw text data using state-of-the-art methods.

	•	BERT Model Training: Fine-tunes a pre-trained BERT model for text classification tasks.

	•	Evaluation: Computes classification metrics such as accuracy, precision, recall, and F1-score on validation data.

	•	Visualization: Generates performance plots and confusion matrices to assess model performance.

	•	TPU Support: Optimized for training on TPU hardware for faster model training.

**Project Highlights**:

	•	Problem: Developing a robust text classifier using BERT to categorize raw text data into predefined labels. This solution can be used in real-world applications such as sentiment analysis, spam detection, or topic classification.
 
	•	Methodology: The project utilizes a pre-trained BERT model, fine-tuned with custom data. Various preprocessing steps, including text cleaning, tokenization, and padding, ensure the input data is suitable for BERT. Hyperparameters are optimized, and the model is trained on TPU for improved performance.
 
	•	Results & Evaluation: The classifier’s performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices. Key visualizations, including training/validation loss curves and confusion matrices, help assess the model’s effectiveness.
 
	•	Lessons Learned: This project highlights the power of pre-trained transformer models and the importance of fine-tuning them for specific tasks. Future improvements could include experimenting with other transformer models or adding more layers to the neural network.

- **Status**: Completed

### [Energy Predict](https://github.com/erenaktuerk/energy-predict)  
A machine learning model designed to predict energy consumption based on temporal factors, such as time of day and temperature. This project emphasizes efficient energy management.

- **Model Type**: Regression
- **Technologies used**: Python, XGBoost, TensorFlow, Pandas, NumPy, Scikit-learn
- **Key features**: Energy consumption prediction, feature engineering, cross-validation
- **Status**: Completed

### [House Price Prediction](https://github.com/erenaktuerk/house-price-prediction)  
A highly optimized machine learning project aimed at predicting house prices using various features such as location, square footage, and overall condition. This project employs a combination of state-of-the-art techniques, including advanced model selection, hyperparameter tuning, and rigorous cross-validation, to deliver accurate, reliable, and interpretable predictions. The solution is designed to be both practically applicable and technically robust, showcasing industry-standard practices for building and deploying machine learning models.

- **Model Type**: Regression (XGBoost, TensorFlow-based MLP)
- **Technologies used**: Python, XGBoost, TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, RandomizedSearchCV, KFold Cross-Validation
- **Key features**: 

	•	Multiple Model Comparison: Implements both XGBoost and TensorFlow-based MLP models to benchmark and identify the best-performing approach for predicting house prices.

	•	Comprehensive Hyperparameter Tuning: Uses RandomizedSearchCV for efficient exploration of hyperparameters, optimizing critical settings such as learning rate, maximum depth, and number of estimators for near-optimal model performance.

	•	Robust Cross-Validation: Employs KFold cross-validation to ensure the model’s stability and generalizability, resulting in reliable performance across diverse datasets.

	•	Model Evaluation and Metrics: Comprehensive evaluation using performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score to deliver a well-rounded model assessment.

	•	Feature Importance Visualization: Visualizes feature importance to understand how individual attributes impact house price predictions, aiding in model transparency and interpretability.

	•	End-to-End Machine Learning Pipeline: A fully integrated pipeline from data preprocessing to model evaluation and visualization, ensuring scalability, ease of use, and practical deployment.

- **Project Highlights**: 

	•	Real-World Applicability: Designed to solve practical house price prediction problems, offering a reliable solution for stakeholders in real estate, investment, and market analysis.

	•	Problem-Solving Oriented: Focuses on overcoming common challenges such as feature selection, model selection, hyperparameter optimization, and ensuring robust generalization across data variations.

	•	High Performance and Interpretability: Achieves strong performance metrics while maintaining transparency, making it suitable for decision-makers who require reliable predictions and an understanding of how the model works.

	•	Scalable and Modular Architecture: Built with scalability in mind, allowing for future improvements such as additional feature engineering, model updates, or even incorporating more complex algorithms like deep learning.

	•	Production-Ready: Well-structured and professionally documented codebase, suitable for deployment in production environments, ensuring ease of maintenance and enhancement.

- **Status**: Completed

## ML API Deployment

### [Credit Risk API](https://github.com/erenaktuerk/credit-risk-api)  
A machine learning model for predicting credit risk, deployed as an API. This project demonstrates the deployment of ML models into production using Flask.

- **Technologies used**: Python, Flask, Scikit-learn, Pandas
- **Key features**: 

	•	Real-Time Risk Assessment: Provides immediate credit risk predictions by processing incoming financial data through the deployed model.

	•	Secure & Scalable Deployment: Utilizes Flask with best practices for API security and scalability, ensuring reliable operation in production environments.

	•	Comprehensive Logging & Monitoring: Features detailed error handling, logging, and performance tracking to facilitate ongoing maintenance and optimization.

- **Status**: Completed

## Python: Exploratory Statistics for Machine Learning

### [Titanic Exploratory Data Analysis](https://github.com/erenaktuerk/Titanic-Exploratory-Data-Analysis)  
An exploratory analysis of the Titanic dataset to gain insights into the data and understand key features for prediction.

- **Technologies used**: Python, Pandas, Matplotlib, Seaborn
- **Key features**: 

	•	In-Depth Data Exploration: Conducts a systematic analysis of missing values, distributions, and variable correlations to reveal hidden patterns.

	•	Advanced Data Visualization: Leverages Matplotlib and Seaborn to create compelling visualizations (e.g., heatmaps, pair plots) that effectively communicate data trends.

	•	Rigorous Hypothesis Testing: Applies statistical tests to validate assumptions and identify significant predictors, laying the groundwork for future predictive modeling.

- **Status**: Completed

## TensorFlow Learning Units

### [Foundations-of-Machine-Learning-with-TensorFlow](https://github.com/erenaktuerk/Foundations-of-Machine-Learning-with-TensorFlow)  
this section introduces foundational machine learning concepts, TensorFlow basics, data preprocessing, and neural network implementation for regression and classification tasks.

- **Technologies used**: Python, TensorFlow
- **Key features**: TensorFlow basics, neural networks, model training
- **Status**: Completed

### [Neural-Network-Classification-in-TensorFlow](https://github.com/erenaktuerk/Neural-Network-Classification-in-TensorFlow)  
This section explores neural network classification in TensorFlow, covering binary/multiclass models, imbalanced data strategies, key evaluation metrics, and deployment via TensorFlow Serving/TFLite.

- **Technologies used**: Python, TensorFlow
- **Key features**: Convolutional neural networks (CNNs), recurrent neural networks (RNNs), model evaluation
- **Status**: Completed
- 
### [Computer-Vision-and-Convolutional-Neural-Networks-CNNs](https://github.com/erenaktuerk/Computer-Vision-and-Convolutional-Neural-Networks-CNNs-)
this section introduces Convolutional Neural Networks (CNNs) and computer vision essentials—covering convolution, pooling, TensorFlow/Keras implementation, transfer learning, and data augmentation for image classification (e.g., MNIST, CIFAR-10).

- **Technologies used**: Python, TensorFlow
- **Key features**: Convolutional neural networks (CNNs), recurrent neural networks (RNNs), model evaluation
- **Status**: Completed

## Python: Restful API

### [To-Do API](https://github.com/erenaktuerk/To_Do_API)  
A simple RESTful API for managing a to-do list, with CRUD operations for adding, updating, and deleting tasks.

- **Technologies used**: Python, Flask, SQLite
- **Key features**: 

	•	Comprehensive CRUD Implementation: Enables robust management of tasks through fully functional Create, Read, Update, and Delete operations.

	•	Scalable RESTful Design: Adheres to REST principles for stateless communication, facilitating seamless integration with various front-end applications.

	•	Efficient Data Persistence: Utilizes SQLite for lightweight data storage, coupled with thorough error handling and validation to ensure data integrity.

- **Status**: Completed
- 
## Web Development - My Projects @RacineDeux
The following only lists projects that I worked on alone as part of my employment. These are reuploads because I worked with a work account as part of my employment. I do not upload group work, which made up the majority of the employment, to my private profile. 

### [Kristall Restaurant - web page](https://github.com/erenaktuerk/kristallweb)

### [Cayan Meisterbetrieb - web page](https://github.com/erenaktuerk/cayanmeisterbetrieb)

### [Net Immobilien - web page](https://github.com/erenaktuerk/netimmobilien)

### [Tiggolino - web page](https://github.com/erenaktuerk/tiggolino)

### [Tumuratas - web page](https://github.com/erenaktuerk/Tumuratas)


## Web Development - Early steps

### [HTML Portfolio](https://github.com/erenaktuerk/html-portfolio)  
A basic personal portfolio website built using HTML and CSS, showcasing web development skills.

- **Technologies used**: HTML, CSS
- **Key features**: Personal portfolio, static website, responsive design
- **Status**: Early project, beginner-level

### [Menu](https://github.com/erenaktuerk/menu)  
A small project to create a responsive menu using HTML, CSS, and JavaScript.

- **Technologies used**: HTML, CSS, JavaScript
- **Key features**: Responsive menu, basic web development practice
- **Status**: Early project, beginner-level

### [Movies](https://github.com/erenaktuerk/Movies)  
A simple movie listing website that fetches data from an API and displays movie information.

- **Technologies used**: HTML, CSS, JavaScript, API integration
- **Key features**: API integration, dynamic content rendering
- **Status**: Early project, beginner-level

### [Movies Pagination](https://github.com/erenaktuerk/movies_pagination)  
A project that implements pagination for movie listings, improving user experience with large datasets.

- **Technologies used**: HTML, CSS, JavaScript
- **Key features**: Pagination, dynamic content loading
- **Status**: Early project, beginner-level

### [Football](https://github.com/erenaktuerk/football)  
A football stats project that collects and displays live football data from various sources.

- **Technologies used**: HTML, CSS, JavaScript, API integration
- **Key features**: Live data, API usage, sports analytics
- **Status**: Early project, beginner-level

### [ErensPage](https://github.com/erenaktuerk/ErensPage)  
A simple personal webpage built to demonstrate HTML and CSS skills with basic design elements.

- **Technologies used**: HTML, CSS
- **Key features**: Personal webpage, static design
- **Status**: Early project, beginner-level

### [ErensPersonalPage](https://github.com/erenaktuerk/erenspersonalpage)  
Another personal webpage project to practice web design and development.

- **Technologies used**: HTML, CSS
- **Key features**: Static personal page, responsive design
- **Status**: Early project, beginner-level

### [CV](https://github.com/erenaktuerk/cv)  
A digital version of a personal CV created with HTML and CSS.

- **Technologies used**: HTML, CSS
- **Key features**: Personal CV, responsive layout
- **Status**: Early project, beginner-level
