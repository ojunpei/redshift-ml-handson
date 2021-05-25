/*
https://archive.ics.uci.edu/ml/datasets/iris

Data Set Information:

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) 
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Predicted attribute: class of iris plant.

This is an exceedingly simple domain.

This data differs from the data presented in Fishers article (identified by Steve Chadwick, spchadwick '@' espeedaz.net ). The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa" where the error is in the fourth feature. The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa" where the errors are in the second and third features.


Attribute Information:

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica */



/* Redshift ML Workshop - Use case II - Data Science User  */

DROP TABLE IF EXISTS iris_data_train;

CREATE TABLE iris_data_train (
  Id int, 
  SepalLengthCm float, 
  SepalWidthCm float,
  PetalLengthCm float, 
  PetalWidthCm float, 
  Species varchar(15)
  );

COPY iris_data_train from 's3://redshift-downloads/redshift-ml/workshop/iris-data/train/' REGION 'us-east-1' IAM_ROLE '<< REPLACE IAM_ROLE >>' CSV IGNOREHEADER 1 ;



DROP TABLE IF EXISTS iris_data_test;

CREATE TABLE iris_data_test (
  Id int, 
  SepalLengthCm float, 
  SepalWidthCm float,
  PetalLengthCm float, 
  PetalWidthCm float, 
  Species varchar(15)
  );

COPY iris_data_test from 's3://redshift-downloads/redshift-ml/workshop/iris-data/test/' REGION 'us-east-1' IAM_ROLE '<< REPLACE IAM_ROLE >>' CSV IGNOREHEADER 1 ;



/* Machine Learning User Example */
/* User creates model and supplies some information like the PROBLEM_TYPE and OBJECTIVE as part of the create model process */ 
/* Create model uses SageMaker Autopilot and chooses specified  PROBLEM_TYPE and OBJECTIVE without trying out other options */


/* PROBLEM_TYPE -  multiclass classification , for all problem_types supported by SageMaker Autopilot - https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development-problem-types.html */ 
/* OBJECTIVE - accuracy ,  for all objectives for xgboost : https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters*/ 

CREATE MODEL model_iris
FROM (
SELECT 
   Id,
   SepalLengthCm,
   SepalWidthCm,
   PetalLengthCm,
   PetalWidthCm,
   Species
FROM iris_data_train
)
TARGET Species 
FUNCTION func_model_iris IAM_ROLE '<< REPLACE IAM_ROLE >>' 
PROBLEM_TYPE multiclass_classification 
OBJECTIVE 'accuracy' 
SETTINGS (S3_BUCKET '<< REPLACE S3 bucket >>');



SELECT json_extract_path_text(modexecmeta, 'train_job_id') FROM pg_ml_model WHERE modname = 'model_iris';


--100 % accuracy 

WITH infer_data AS (
    SELECT Species AS label,
        func_model_iris(Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) AS predicted,
        CASE WHEN label is NULL THEN NULL ELSE label END AS actual,
        CASE WHEN actual = predicted THEN 1::INT
        ELSE 0::INT END AS correct
    FROM iris_data_test
),
aggr_data AS (
    SELECT SUM(correct) as num_correct, COUNT(*) as total FROM infer_data
)
SELECT (num_correct::float/total::float) AS accuracy FROM aggr_data;




--Predict the class of iris flower 
WITH class_data AS ( SELECT func_model_iris( 
   Id,
   SepalLengthCm,
   SepalWidthCm,
   PetalLengthCm,
   PetalWidthCm) AS class 
FROM iris_data_test )
SELECT 
CASE WHEN class = 'Iris-versicolor'  THEN 'Class-Iris-versicolor'
     WHEN class = 'Iris-setosa'  THEN 'Class-Iris-setosa'
     WHEN class = 'Iris-virginica'  THEN 'Class-Iris-virginica'
     ELSE 'Class-Other' END as class_distribution,
COUNT(1) AS count
from class_data GROUP BY 1;

/*   Sample output

dev-# from class_data GROUP BY 1;
  class_distribution   | count
-----------------------+-------
 Class-Iris-versicolor |    82
 Class-Iris-setosa     |    81
 Class-Iris-virginica  |    88
(3 rows)

*/

--Only for live workshop

CREATE MODEL model_iris_v2
FROM (
SELECT 
   Id,
   SepalLengthCm,
   SepalWidthCm,
   PetalLengthCm,
   PetalWidthCm,
   Species
FROM iris_data_train
)
TARGET Species 
FUNCTION func_model_iris_v2 IAM_ROLE '<< REPLACE IAM_ROLE >>' 
PROBLEM_TYPE multiclass_classification 
OBJECTIVE 'accuracy' 
SETTINGS (S3_BUCKET '<< REPLACE S3 bucket >>', MAX_RUNTIME 1800);







