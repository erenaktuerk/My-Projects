# My-Projects

Collection and categorization of my projects

Overview:
# Machine Learning Projects
- [GCP-Image-CLassifier](#GCP-Image-Classifier) (in Progress)
- [CNN CIFAR-10 Image Classifier](#CNN-CIFAR-10-IMAGE-CLASSIFIER) (Completed)
- [AAPL Share Prediction](#AAPL-Share-Prediction) (Completed)
- [BERT Text Classifier](#BERT-Text-CLassifier) (Completed)
- [Energy Predict](#Energy-Predict) (Completed)
- [House Price Prediction](#House-Price-Prediction)
- [Credit Risk API](#Credit-Risk-API)
- [Titanic Exploratory Data Analysis](#Titanic-Epxloratory-Data-Analysis)

# TensorFlow Learning Units
- [Foundations-of-Machine-Learning-with-TensorFlow](#Foundations-of-Machine-Learning-with-TensorFlow)
- [Neural-Network-Classification-in-TensorFlow](#Neural-Network-Classification-in-TensorFlow)
- [Computer-Vision-and-Convolutional-Neural-Networks-CNNs](#Computer-Vision-and-Convolutional-Neural-Networks-CNNs)

# Python: Restful API
- [To-Do API](#To-Do API)

# Web Development - My Projects @RacineDeux
- [Kristall Restaurant - web page](#Kristall-Restaurant-web-page)
- [Cayan Meisterbetrieb - web page](#Cayan-Meisterbetrieb-web-page)
- [Net Immobilien - web page](#Net-Immobilien-web-page)
- [Tiggolino - web page](#Tiggolino-web-page)
- [Tumuratas - web page](#Tumuratas-web-page)

## Machine Learning Projects

### [GCP-Image-Classifier](https://github.com/erenaktuerk/gcp-image-classifier)
A cloud-based image classification project using deep learning. This system utilizes a Convolutional Neural Network (CNN) for high-performance image recognition and is deployed on Google Cloud Platform (GCP) for scalability. The model is trained on Google AI Platform, with data stored in Google Cloud Storage (GCS), ensuring efficient and reliable cloud-based operations.

**Technologies Used**
	•	Frameworks & Libraries: Python, TensorFlow, Keras, OpenCV, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
	•	Cloud Services: Google Cloud Storage (GCS), Google AI Platform (Vertex AI), Google Compute Engine, Google Cloud SDK
	•	Additional Tools: Keras Tuner for hyperparameter optimization, Grad-CAM for model interpretability

**Key Features**

	•	Cloud-Based Model Training: Utilizes Google AI Platform to train deep learning models efficiently with scalable cloud infrastructure.
 
	•	Automated Data Processing: Image preprocessing and augmentation are handled dynamically using OpenCV and TensorFlow Image Preprocessing.
 
	•	Interpretable AI: Implements Grad-CAM to visualize which image features influence predictions.

**Project Highlights**

	•	Seamless Google Cloud Integration: The pipeline is fully optimized for GCP, from data storage to model training and deployment.
 
	•	Optimized Deep Learning Architecture: Uses Batch Normalization, Dropout, and data augmentation to improve generalization and prevent overfitting.
 
	•	Scalable Deployment: Deployed on Vertex AI, allowing real-time inference and easy scalability for production environments.

**Status**: In Progress

### [CNN CIFAR-10 Image Classifier](https://github.com/erenaktuerk/cnn-cifar10-classifier)

A deep learning-based image classification project leveraging a Convolutional Neural Network (CNN). This solution integrates hyperparameter optimization, data augmentation, and model interpretability to enhance performance and generalization. Designed for both research and real-world deployment, the project achieves high accuracy on CIFAR-10.

**Model Type**: Classification

**Technologies Used**: Frameworks & Libraries: Python, TensorFlow, Keras, Keras Tuner, OpenCV, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

**Key Features**

	•	Optimized CNN Architecture: Incorporates Batch Normalization and Dropout layers to minimize overfitting and enhance feature extraction.
 
	•	Hyperparameter Optimization: Uses Keras Tuner to automatically optimize key parameters for improved model accuracy.
 
	•	Model Interpretability: Implements Grad-CAM and feature map visualizations to explain decision-making.

**Project Highlights**

	•	Real-World Applications: Designed for practical scenarios, ensuring reliability and high performance.
 
	•	Problem-Solving Focus: Effectively addresses challenges like overfitting, hyperparameter selection, and model transparency.
 
	•	Scalable Architecture: Built with flexibility in mind, allowing easy integration of enhancements and future improvements.
 
**Status**: Completed

### [AAPL Share Prediction](https://github.com/erenaktuerk/aapl-share-prediction)  
A machine learning-based regression model designed to predict Apple Inc. (AAPL) stock prices using historical market data and temporal features. The project focuses on feature engineering, data preprocessing, and hyperparameter optimization to enhance predictive accuracy. Both XGBoost and a TensorFlow-based MLP are evaluated to determine the best-performing model.

**Model Type**: Regression

**Technologies Used**: Frameworks & Libraries: Python, XGBoost, TensorFlow, Pandas, NumPy, Scikit-learn

**Key Features**

	•	Advanced Data Preprocessing: Cleans and selects key features to improve prediction quality.
 
	•	Feature Engineering: Extracts temporal trends to capture market fluctuations.
 
	•	Hyperparameter Optimization: Utilizes RandomizedSearchCV to fine-tune models for maximum accuracy.

**Project Highlights**

	•	Problem-Solving Approach: Predicts future AAPL stock prices to aid investment decisions.
 
	•	Model Evaluation: Compares XGBoost vs. TensorFlow MLP, using RMSE and MAE for performance assessment.
 
	•	Robust Validation: Employs cross-validation to prevent overfitting and ensure reliability.
 
**Status**: Completed

### [BERT Text Classifier](https://github.com/erenaktuerk/bert-text-classifier)  
This project applies BERT (Bidirectional Encoder Representations from Transformers) for text classification, specifically focusing on categorizing user reviews into sentiment or topic-based classes. Using BERT’s bidirectional approach for deep contextual understanding, the project integrates advanced NLP techniques, including data preprocessing, model fine-tuning, and performance evaluation. The model is optimized for TPU hardware for faster training and superior performance.

**Model Type**: CLassification

**Technologies Used**: Frameworks & Libraries: Python, BERT, HuggingFace, PyTorch, Scikit-learn

**Key Features**

	•	Text Preprocessing: Implements cutting-edge methods for cleaning and tokenizing text data.
 
	•	BERT Model Training: Fine-tunes a pre-trained BERT model tailored for text classification.
 
	•	Evaluation: Assesses model performance using metrics like accuracy, precision, recall, and F1-score.

**Project Highlights**

	•	Problem-Solving Approach: Uses BERT for robust text classification in real-world applications, such as sentiment analysis or topic classification.
 
	•	Model Optimization: Incorporates TPU hardware to speed up training, improving model efficiency.
 
	•	Evaluation Metrics: Uses confusion matrices and performance plots to visualize and evaluate model effectiveness.

**Status**: Completed

### [Energy Predict](https://github.com/erenaktuerk/energy-predict)  
This machine learning project focuses on predicting energy consumption based on temporal factors such as time of day and temperature. The goal is to optimize energy management by accurately forecasting energy demand, making it more efficient and adaptable to real-world scenarios.

**Model Type**: Regression
**Technologies Used**: Frameworks & Libraries: Python, XGBoost, TensorFlow, Pandas, NumPy, Scikit-learn

**Key Features**

	•	Advanced Feature Engineering: Utilizes economic and demographic indicators to enhance data for better prediction accuracy.
 
	•	Model Benchmarking: Compares XGBoost and TensorFlow-based MLP to select the best model based on performance.
 
	•	Hyperparameter Tuning: Implements RandomizedSearchCV for efficient model parameter optimization.

**Project Highlights**

	•	Real-World Focus: Aims to solve practical energy consumption prediction challenges with reliable results.
 
	•	Problem-Solving Approach: Applies sophisticated data manipulation and feature engineering techniques for enhanced model performance.
 
	•	High Performance & Scalability: Ensures a robust, flexible, and scalable architecture for production-level deployment.
 
**Status**: Completed

### [House Price Prediction](https://github.com/erenaktuerk/house-price-prediction)  
A practical machine learning project that predicts house prices based on factors like location, size, and condition. It uses model selection, hyperparameter tuning, and cross-validation to provide accurate, interpretable results.
**Model Type**: Regression
**Technologies used**: Python, XGBoost, TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, RandomizedSearchCV, KFold Cross-Validation

**Key features**:

	•	Model benchmarking with XGBoost and TensorFlow-based MLP.
 
	•	Hyperparameter tuning using RandomizedSearchCV and cross-validation.
 
	•	Comprehensive evaluation with RMSE, MAE, MSE, R², and feature importance visualization.

**Project Highlights**:

	•	Solves practical house price prediction challenges with high accuracy.
 
	•	Focuses on model optimization, feature selection, and generalization.
 
	•	Balances high performance with interpretability and scalability.
 

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
- 
- **Key features**: TensorFlow basics, neural networks, model training
- 
- **Status**: Completed

### [Neural-Network-Classification-in-TensorFlow](https://github.com/erenaktuerk/Neural-Network-Classification-in-TensorFlow)  
This section explores neural network classification in TensorFlow, covering binary/multiclass models, imbalanced data strategies, key evaluation metrics, and deployment via TensorFlow Serving/TFLite.

- **Technologies used**: Python, TensorFlow
- 
- **Key features**: Convolutional neural networks (CNNs), recurrent neural networks (RNNs), model evaluation
- 
- **Status**: Completed
- 
### [Computer-Vision-and-Convolutional-Neural-Networks-CNNs](https://github.com/erenaktuerk/Computer-Vision-and-Convolutional-Neural-Networks-CNNs-)
this section introduces Convolutional Neural Networks (CNNs) and computer vision essentials—covering convolution, pooling, TensorFlow/Keras implementation, transfer learning, and data augmentation for image classification (e.g., MNIST, CIFAR-10).

- **Technologies used**: Python, TensorFlow
- 
- **Key features**: Convolutional neural networks (CNNs), recurrent neural networks (RNNs), model evaluation
- 
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


## Web Development - My Projects @RacineDeux
The following only lists projects that I worked on alone as part of my employment. These are reuploads because I worked with a work account as part of my employment. I do not upload group work, which made up the majority of the employment, to my private profile. 

### [Kristall Restaurant - web page](https://github.com/erenaktuerk/kristallweb)
Kristall Web is a modern, responsive web application designed specifically for Restaurant Kristall. The project aims to deliver an engaging and intuitive online experience, offering visitors a comprehensive view of the restaurant’s offerings—from the menu and reservation system to contact information—through a visually appealing and user-friendly interface.

### [Cayan Meisterbetrieb - web page](https://github.com/erenaktuerk/cayanmeisterbetrieb)
Cayan Meisterbetrieb is a professional web application designed for a car workshop, providing a comprehensive and user-friendly online presence. The website showcases the workshop’s services, company history, and contact information in a modern, responsive design. This README provides an in-depth analysis of the project, covering its features, technologies used, setup instructions, and guidelines for customization and contribution.

### [Net Immobilien - web page](https://github.com/erenaktuerk/netimmobilien)
NetImmobilien is a modern web application designed for a real estate agency. It provides a comprehensive platform where users can browse property listings, view detailed real estate information, and contact agents directly. The project combines an elegant design with robust functionality to create an engaging online presence for the agency.

### [Tiggolino - web page](https://github.com/erenaktuerk/tiggolino)
This repository contains the source code for the *Tiggolino Children's Playground* website. The project is developed using the *Django* web framework and serves as the online presence for the Tiggolino children's recreational park.

### [Tumuratas - web page](https://github.com/erenaktuerk/Tumuratas)
This repository contains the web project for *Tumuratas Group, built using the **Django* web framework.

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
