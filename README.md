# My-Projects

Collection and categorization of my projects

## Machine Learning Projects

### [CNN CIFAR-10 Image Classifier](https://github.com/erenaktuerk/cnn-cifar10-classifier)
A professionally designed image classification solution using an advanced Convolutional Neural Network (CNN) architecture. This project incorporates a robust deep learning pipeline with hyperparameter optimization, data augmentation, and model interpretability, focusing on real-world applications and deployment.
- **Model Type**: Convolutional Neural Network (CNN)
- **Technologies Used**: Python, TensorFlow, Keras, Keras Tuner, OpenCV, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Key Features**:
  
	•	Optimized CNN Architecture: Deep feature extraction with minimal overfitting, using Batch Normalization and Dropout layers.

	•	Efficient Hyperparameter Tuning: Leverages Keras Tuner for optimizing key parameters to maximize model performance.

	•	Data Augmentation: Enhances model generalization through real-time augmentation techniques.

	•	Model Interpretability: Utilizes feature map visualizations and Grad-CAM to explain model decisions.

	•	Modular Pipeline: A flexible, end-to-end pipeline suitable for both research and production environments.

- **Project Highlights**:

	•	Real-World Applications: Focused on reliability and performance in practical settings.

	•	Problem-Solving Oriented: Addresses common deep learning issues such as overfitting, hyperparameter selection, and model transparency, making it a comprehensive solution for complex computer vision tasks.

	•	Problem-Solving Approach: Addresses challenges like overfitting and hyperparameter selection.

	•	High Performance: Achieves competitive accuracy on CIFAR-10 with thorough evaluation.

	•	Scalable Architecture: Designed to easily integrate enhancements and future improvements.
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
A practical and performance-driven machine learning project for house price prediction, using features like location, square footage, and condition. By combining advanced model selection, hyperparameter tuning, and cross-validation, it delivers accurate and interpretable results, following industry best practices.

- **Model Type**: Regression (XGBoost, TensorFlow-based MLP)
- **Technologies used**: Python, XGBoost, TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, RandomizedSearchCV, KFold Cross-Validation
- **Key features**: 

	•	Model Benchmarking: Compares XGBoost and TensorFlow-based MLP to select the best-performing approach.

	•	Hyperparameter Tuning: Uses RandomizedSearchCV to efficiently optimize key parameters.

	•	Robust Cross-Validation: KFold cross-validation ensures stability and generalization.

	•	Comprehensive Evaluation: Assesses performance with RMSE, MAE, MSE, and R² scores.

	•	Feature Importance: Visualizes key factors influencing price predictions for better interpretability.

	•	End-to-End Pipeline: Covers data preprocessing, model training, and evaluation for seamless deployment.

- **Project Highlights**: 

	•	Real-World Focus: Solves practical house price prediction challenges with reliable performance.

	•	Problem-Solving Oriented: Addresses feature selection, model optimization, and generalization.

	•	High Performance & Transparency: Balances strong metrics with interpretability.

	•	Scalable Design: Easily adaptable for future improvements and feature expansion.
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
