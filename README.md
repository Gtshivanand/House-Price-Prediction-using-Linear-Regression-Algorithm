#  🏡 House Price Prediction Analysis Using Linear Regression Algorithm 🏡

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) 
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) 
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  
[![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io)
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

<img src="https://github.com/Gtshivanand/House-Price-Prediction-using-Linear-Regression-Algorithm/blob/main/images/GrayBrickHouse-social-share.jpg"/>

## Introduction:

Accurately predicting house prices is crucial for both buyers and sellers in the real estate market. This project utilizes the Linear Regression Algorithm to forecast house prices based on key features such as property size, location, and other critical attributes. By analyzing these factors, the model helps stakeholders make data-driven decisions and understand the variables influencing property values.

## Project Overview:

The **House Price Prediction** project aims to predict the selling price of houses based on various features using the **Linear Regression Algorithm**. This analysis helps property investors, sellers, and buyers to accurately estimate house prices and make informed decisions.

---

## Problem Statement:

A key challenge for property sellers is determining the sale price of a property. The ability to predict the exact property value is beneficial for property investors as well as buyers to plan their finances according to the price trend. The property prices depend on several features like:

- Property area
- Basement square footage
- Year built
- Number of bedrooms
- Road access type

By using **Linear Regression**, we aim to build a model that can accurately predict house prices based on these factors.

---

## Dataset Information

The dataset used for this analysis contains detailed information about houses, including their physical attributes and location details. Key features include:

## Data Definition:

- **Dwell_Type:** Identifies the type of dwelling involved in the sale

- **Zone_Class:** Identifies the general zoning classification of the sale
	
- **LotFrontage:** Linear feet of street-connected to the property

- **LotArea:** Lot size is the lot or parcel side where it adjoins a street, boulevard or access way

- **Road_Type:** Type of road access to the property
       	
- **Alley:** Type of alley access to the property
		
- **Property_Shape:** General shape of the property

- **LandContour:** Flatness of the property

- **LotConfig:** Lot configuration
	
- **LandSlope:** Slope of property

- **Neighborhood:** Physical locations within Ames city limits
			
- **Condition1:** Proximity to various conditions

- **Condition2:** Proximity to various conditions (if more than one is present)
	
- **Dwelling_Type:** Type of dwelling
	
- **HouseStyle:** Style of dwelling

- **OverallQual:** Rates the overall material and finish of the house
	
- **OverallCond:** Rates the overall condition of the house
		
- **YearBuilt:** Original construction date

- **YearRemodAdd:** Remodel date (same as construction date if no remodeling or additions)

- **RoofStyle:** Type of roof

- **RoofMatl:** Roof material
		
- **Exterior1st:** Exterior covering on the house
	
- **Exterior2nd:** Exterior covering on the house (if more than one material)

- **MasVnrType:** Masonry veneer type

- **MasVnrArea:** Masonry veneer area in square feet

- **ExterQual:** Evaluates the quality of the material on the exterior

- **ExterCond:** Evaluates the present condition of the material on the exterior
		
- **Foundation:** Type of foundation
		
- **BsmtQual:** Evaluates the height of the basement
		
- **BsmtCond:** Evaluates the general condition of the basement
	
- **BsmtExposure:** Refers to walkout or garden level walls

- **BsmtFinType1:** Rating of basement finished area
		
- **BsmtFinSF1:** Type 1 finished square feet

- **BsmtFinType2:** Rating of basement finished area (if multiple types)

- **BsmtFinSF2:** Type 2 finished square feet

- **BsmtUnfSF:** Unfinished square feet of the basement area

- **TotalBsmtSF:** Total square feet of the basement area

- **Heating:** Type of heating
		
- **HeatingQC:** Heating quality and condition
		
- **CentralAir:** Central air conditioning

- **Electrical:** Electrical system
		
- **1stFlrSF:** First Floor square feet
 
- **2ndFlrSF:** Second floor square feet

- **LowQualFinSF:** Low quality finished square feet (all floors)

- **GrLivArea:** Above grade (ground) living area square feet

- **BsmtFullBath:** Basement full bathrooms

- **BsmtHalfBath:** Basement half bathrooms

- **FullBath:** Full bathrooms above grade

- **HalfBath:** Half baths above grade

- **Bedroom:** Bedrooms above grade (does NOT include basement bedrooms)

- **Kitchen:** Kitchens above grade

- **KitchenQual:** Kitchen quality

- **TotRmsAbvGrd:** Total rooms above grade (does not include bathrooms)

- **Functional:** Home functionality (Assume typical unless deductions are warranted)

- **Fireplaces:** Number of fireplaces

- **FireplaceQu:** Fireplace quality

- **GarageType:** Garage location
		
- **GarageYrBlt:** Year garage was built
		
- **GarageFinish:** Interior finish of the garage

- **GarageCars:** Size of garage in car capacity

- **GarageArea:** Size of garage in square feet

- **GarageQual:** Garage quality
		
- **GarageCond:** Garage condition
		
- **PavedDrive:** Paved driveway
		
- **WoodDeckSF:** Wood deck area in square feet

- **OpenPorchSF:** Open porch area in square feet

- **EnclosedPorch:** Enclosed porch area in square feet

- **3SsnPorch:** Three season porch area in square feet

- **ScreenPorch:** Screen porch area in square feet

- **PoolArea:** Pool area in square feet

- **PoolQC:** Pool quality
		
- **Fence:** Fence quality
		
- **MiscFeature:** Miscellaneous feature not covered in other categories
		
- **MiscVal:** Value of miscellaneous feature

- **MoSold:** Month Sold (MM)

- **YrSold:** Year Sold (YYYY)

- **SaleType:** Type of sale

- **SaleCondition:** Condition of sale
       
- **Property_Sale_Price:** Price of the house
---

## Project Workflow

1. **Data Collection:**
   
   - Load the dataset and explore its structure.

3. **Data Preprocessing:**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling and normalization

4. **Exploratory Data Analysis (EDA):**
   - Visualize relationships between features and target variable
   - Identify key patterns and outliers

5. **Model Building:**
   - Split data into training and test sets
   - Train a **Linear Regression** model

6. **Model Evaluation:**
   - Evaluate model performance using metrics such as:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - R-squared score

7. **Prediction & Interpretation:**
   - Make predictions on unseen data
   - Interpret the influence of features on house prices

---

## Dependencies

Ensure the following Python libraries are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
## Table of Contents:

1. **[Import Libraries](#import_lib)**
2. **[Set Options](#set_options)**
3. **[Read Data](#Read_Data)**
4. **[Prepare and Analyze the Data](#data_preparation)**
    - 4.1 - [Understand the Data](#Data_Understanding)
        - 4.1.1 - [Data Type](#Data_Types)
        - 4.1.2 - [Summary Statistics](#Summary_Statistics)
        - 4.1.3 - [Distribution of Variables](#distribution_variables)
        - 4.1.4 - [Discover Outliers](#outlier)
        - 4.1.5 - [Missing Values](#Missing_Values)
        - 4.1.6 - [Correlation](#correlation)
        - 4.1.7 - [Analyze Relationships Between Target and Categorical Variables](#cat_num)
    - 4.2 - [Data Preparation](#Data_Preparation)
        - 4.2.1 - [Check for Normality](#Normality)
        - 4.2.2 - [Dummy Encode the Categorical Variables](#dummy)
5. **[Linear Regression (OLS)](#LinearRegression)**
    - 5.1 - [Multiple Linear Regression Full Model with Log Transformed Dependent Variable (OLS)](#withLog)
    - 5.2 - [Multiple Linear Regression Full Model without Log Transformed Dependent Variable (OLS)](#withoutLog)
    - 5.3 - [Feature Engineering](#Feature_Engineering)
      - 5.3.1 - [Multiple Linear Regression (Using New Feature) - 1](#feature1)
      - 5.3.2 - [Multiple Linear Regression (Using New Feature) - 2](#feature2)
6. **[Feature Selection](#feature_selection)**
     - 6.1 - [Variance Inflation Factor](#vif)

7. **[Conclusion and Interpretation](#conclusion)**

### Libraries Used:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Usage Instructions:

1. Clone the repository:

```bash
git clone https://github.com/https://Gtshivanand/House-Price-Prediction-using-Linear-Regression-Algorithm
```

2. Run the Jupyter Notebook:

```bash
jupyter notebook House_Price_Prediction.ipynb
```

---

## Results:

- Successfully built a **Linear Regression** model to predict house prices.
- Evaluated model performance using error metrics (MAE, MSE, R-squared).
- Identified key features influencing property prices.

---

## Future Enhancements:

- Implement advanced models like **Random Forest** or **XGBoost** for better accuracy.
- Optimize feature selection and hyperparameters.
- Deploy the model using **Flask** or **Streamlit** for real-time predictions.

---
## Conclusion:

The House Price Prediction project successfully demonstrates the application of Linear Regression in predicting real estate prices. By analyzing essential features, the model provides accurate predictions and valuable insights into the factors affecting house prices. This model can assist stakeholders in making informed decisions and serves as a foundation for future enhancements using advanced machine learning algorithms.

The project highlights the importance of data preprocessing, exploratory analysis, and model evaluation in building a reliable predictive model. With further improvements, this model can be deployed in real-world scenarios to provide dynamic and precise price estimations.


## 📧  Feedback and Suggestions:

Thank you for visiting my repository! If you have any questions or feedback, feel free to reach out.

I’d love to hear your thoughts, feedback, and suggestions! Feel free to connect with me:

 LinkedIn: [Shivanand Nashi](https://www.linkedin.com/in/shivanand-s-nashi-79579821a)
 
 Email: shivanandnashi97@gmail.com


Looking forward to connecting and exchanging ideas!

## ✨ Support this project!
If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!
Your support helps keep the project active and encourages further development.

Thank you for your support! 💖



