/*
https://archive.ics.uci.edu/ml/datasets/bank+marketing

Data Set Information:

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.


The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).


Attribute Information:

Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
*/




/* Redshift ML Workshop - Use case I - Data Analyst User  */


CREATE TABLE bank_details_training(
   age numeric,
   job varchar,
   marital varchar,
   education varchar,
   "default" varchar,
   housing varchar,
   loan varchar,
   contact varchar, 
   month varchar,
   day_of_week varchar,
   duration numeric,
   campaign numeric,
   pdays numeric,
   previous numeric,
   poutcome varchar,
   emp_var_rate numeric,
   cons_price_idx numeric,     
   cons_conf_idx numeric,     
   euribor3m numeric,
   nr_employed numeric,
   y boolean ) ;

COPY bank_details_training from 's3://redshift-downloads/redshift-ml/workshop/bank-marketing-data/training_data/' REGION 'us-east-1' IAM_ROLE '<< REPLACE IAM_ROLE >> ' CSV IGNOREHEADER 1 delimiter ';';



CREATE TABLE bank_details_inference(
   age numeric,
   job varchar,
   marital varchar,
   education varchar,
   "default" varchar,
   housing varchar,
   loan varchar,
   contact varchar, 
   month varchar,
   day_of_week varchar,
   duration numeric,
   campaign numeric,
   pdays numeric,
   previous numeric,
   poutcome varchar,
   emp_var_rate numeric,
   cons_price_idx numeric,     
   cons_conf_idx numeric,     
   euribor3m numeric,
   nr_employed numeric,
   y boolean ) ;

COPY bank_details_inference from 's3://redshift-downloads/redshift-ml/workshop/bank-marketing-data/inference_data/' REGION 'us-east-1' IAM_ROLE '<< REPLACE IAM_ROLE >>' CSV IGNOREHEADER 1 delimiter ';';


/* Complete Autopilot generated with minimal user inputs */
/* This will be a binary classification problem but auto pilot will choose the relevant algorithm based on the data and inputs */

DROP MODEL model_bank_marketing;
CREATE MODEL model_bank_marketing
FROM (
SELECT    
   age ,
   job ,
   marital ,
   education ,
   "default" ,
   housing ,
   loan ,
   contact , 
   month ,
   day_of_week ,
   duration ,
   campaign ,
   pdays ,
   previous ,
   poutcome ,
   emp_var_rate ,
   cons_price_idx ,     
   cons_conf_idx ,     
   euribor3m ,
   nr_employed ,
   y 
FROM
    bank_details_training )
    TARGET y
FUNCTION func_model_bank_marketing
IAM_ROLE '<< REPLACE IAM_ROLE >>'
SETTINGS (
  S3_BUCKET '<< REPLACE S3 bucket >>'
  )
;

--Inference/Accuracy on inference dats 

WITH infer_data
 AS (
    SELECT  y as actual, func_model_bank_marketing(age,job,marital,education,"default",housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed) AS predicted,
     CASE WHEN actual = predicted THEN 1::INT
         ELSE 0::INT END AS correct
    FROM bank_details_inference
    ),
 aggr_data AS (
     SELECT SUM(correct) as num_correct, COUNT(*) as total FROM infer_data
 )
 SELECT (num_correct::float/total::float) AS accuracy FROM aggr_data;



--Predict how many will subscribe for term deposit vs not subscribe

WITH term_data AS ( SELECT func_model_bank_marketing( age,job,marital,education,"default",housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed) AS predicted 
FROM bank_details_inference )
SELECT 
CASE WHEN predicted = 'Y'  THEN 'Yes-will-do-a-term-deposit'
     WHEN predicted = 'N'  THEN 'No-term-deposit'
     ELSE 'Neither' END as deposit_prediction,
COUNT(1) AS count
from term_data GROUP BY 1;

/*   Sample output 

     deposit_prediction     | count
----------------------------+-------
 Yes-will-do-a-term-deposit |  5362
 No-term-deposit            | 35826
(2 rows)
*/



/* Only for live workshop */

  CREATE MODEL model_bank_marketing_v2
  FROM (
  SELECT    
     age ,
     job ,
     marital ,
     education ,
     "default" ,
     housing ,
     loan ,
     contact , 
     month ,
     day_of_week ,
     duration ,
     campaign ,
     pdays ,
     previous ,
     poutcome ,
     emp_var_rate ,
     cons_price_idx ,     
     cons_conf_idx ,     
     euribor3m ,
     nr_employed ,
     y 
  FROM
      bank_details_training )
      TARGET y
  FUNCTION func_model_bank_marketing_v2
  IAM_ROLE '< REPLACE IAM_ROLE >>'
  SETTINGS (
    S3_BUCKET '<< REPLACE S3 bucket >>',
    MAX_RUNTIME 1800
    )
  ;






