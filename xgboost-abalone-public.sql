/*
https://archive.ics.uci.edu/ml/datasets/Abalone
Data Set Information:

Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and location (hence food availability) may be required to solve the problem.

From the original data examples with missing values were removed (the majority having the predicted value missing), and the ranges of the continuous values have been scaled for use with an ANN (by dividing by 200).


Attribute Information:

Given is the attribute name, attribute type, the measurement unit and a brief description. The number of rings is the value to predict: either as a continuous value or as a classification problem.

Name / Data Type / Measurement Unit / Description
-----------------------------
Sex / nominal / -- / M, F, and I (infant)
Length / continuous / mm / Longest shell measurement
Diameter / continuous / mm / perpendicular to length
Height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous / grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
Rings / integer / -- / +1.5 gives the age in years

The readme file contains attribute statistics.
*/



/* Redshift ML Workshop - Use case III - Machine Learning Expert User - uses XGBOOST with AUTO OFF */


DROP TABLE IF EXISTS abalone_xgb_train;

CREATE TABLE abalone_xgb_train (
length_val float, 
diameter float, 
height float,
whole_weight float, 
shucked_weight float, 
viscera_weight float,
shell_weight float, 
rings int
);

COPY abalone_xgb_train FROM 's3://redshift-downloads/redshift-ml/workshop/xgboost_abalone_data/train/' REGION 'us-east-1' IAM_ROLE '<< REPLACE IAM_ROLE >>' IGNOREHEADER 1 CSV;





DROP TABLE IF EXISTS abalone_xgb_test;

CREATE TABLE abalone_xgb_test (
length_val float, 
diameter float, 
height float,
whole_weight float, 
shucked_weight float, 
viscera_weight float,
shell_weight float, 
rings int
);

COPY abalone_xgb_test FROM 's3://redshift-downloads/redshift-ml/workshop/xgboost_abalone_data/test/' REGION 'us-east-1' IAM_ROLE '<< REPLACE IAM_ROLE >>' IGNOREHEADER 1 CSV;





/* create a new model with XGBOOST using abalone dataset 
  Replace S3_BUCKET option with your own bucket and the IAM_ROLE with your own IAM_ROLE with necessary privileges including S3 Read/Write and Amazon SageMaker */ 

/* Once the create model command is finished, a function named 'ml_fn_abalone_xgboost_multi_predict_age' which will be used in the subsequent inference (prediction) query below */

/* create model using modified abalone data set as the training data set with record number
   less than 2500 */





drop model model_abalone_xgboost_regression;

-- ~ 10 mins 
CREATE MODEL model_abalone_xgboost_regression 
FROM (SELECT
      length_val,
      diameter,
      height,
      whole_weight,
      shucked_weight,
      viscera_weight,
      shell_weight,
      rings
     FROM abalone_xgb_train)
TARGET Rings 
FUNCTION func_model_abalone_xgboost_regression 
IAM_ROLE '<< REPLACE IAM_ROLE >>' 
AUTO OFF 
MODEL_TYPE xgboost 
OBJECTIVE 'reg:squarederror' 
PREPROCESSORS 'none' 
HYPERPARAMETERS DEFAULT EXCEPT (NUM_ROUND '100') 
SETTINGS (S3_BUCKET '<< REPLACE S3 bucket >>');



-- MSE/RMSE [The lower the better]: For regression problems, we compute Mean Squared Error / Root Mean Squared Error.
--Check accuracy 

WITH infer_data AS (
    SELECT Rings AS label, func_model_abalone_xgboost_regression(
Length_val, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight,
Shell_weight
) AS predicted,
    CASE WHEN label is NULL THEN 0 ELSE label END AS actual
    FROM abalone_xgb_test
)
SELECT SQRT(AVG(POWER(actual - predicted, 2))) AS rmse FROM infer_data;




--Predict the age group of Abalone Species for harvesting, run on the test table
WITH age_data AS ( SELECT func_model_abalone_xgboost_regression( length_val, 
                                               diameter, 
                                               height, 
                                               whole_weight, 
                                               shucked_weight, 
                                               viscera_weight, 
                                               shell_weight ) + 1.5 AS age
FROM abalone_xgb_test )
SELECT 
CASE WHEN age  > 20 THEN 'age_over_20'
     WHEN age  > 10 THEN 'age_between_10_20'
     WHEN age  > 5  THEN 'age_between_5_10'
     ELSE 'age_5_and_under' END as age_group,
COUNT(1) AS count
from age_data GROUP BY 1;


/* Sample output

dev-# from age_data GROUP BY 1;
     age_group     | count
-------------------+-------
 age_between_10_20 |   589
 age_between_5_10  |   247
 age_5_and_under   |     1
 age_over_20       |     1
(4 rows)

*/



