#%%%
# First let's create our PySpark instance
# import findspark
# findspark.init()
#%%
""
from utils.const import TYPE_CAST_LOOKUP, SETTINGS_BASEDIR, SETTINGS_FNAME,DATA_SET_FNAME,DATA_SET_BASEDIR,SCHEMA_BASEDIR,DATA_SET_FNAME,SCHEMA_FNAME
#%%
import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import  StringType, IntegerType, FloatType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
import json
# from pyspark.sql.types import * 
from pyspark.sql.functions import *
# May take awhile locally
#%%ITEM_ID
conf = pyspark.SparkConf()
conf.setAll(
    [('spark.executor.cores',"4")]
)

# %%
def load_json(base_dir, fpath):
    with open(f"{base_dir}{fpath}", "rb") as outfile:
        return json.load(outfile)

_settings = load_json(SETTINGS_BASEDIR,SETTINGS_FNAME)
#%%
sc = pyspark.SparkContext(conf= conf)
spark = SparkSession.builder.appName("Classifier_app").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark
# Click the hyperlinked "Spark UI" link to view details about your Spark session
# %%
df = spark.read.csv(DATA_SET_BASEDIR+DATA_SET_FNAME,inferSchema=True,header=True)

#%%
def null_value_calc(df):
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if(nullRows > 0):
            temp = k,nullRows,(nullRows/numRows)*100
            null_columns_counts.append(temp)
    return(null_columns_counts)

null_columns_calc_list = null_value_calc(df)
if len(null_columns_calc_list) > 0:
    spark.createDataFrame(null_columns_calc_list, ['Column_Name', 'Null_Values_Count','Null_Value_Percent']).show()
#%%
# df = df.na.fill(0)
# df = df.na.fill('0')
# df.printSchema()
data_set_schema = [{"name":column, "type":TYPE_CAST_LOOKUP.get(str(df.schema[column].dataType))} for column in df.columns ]
# %%
def TypeConvetion(data_set_schema, df):
    for type_dict in data_set_schema:
        col,type = type_dict["name"], type_dict["type"]
        if type == "string":
            df = df.withColumn(col, df[col].cast(StringType())) #Rename and change to string type
            df = df.fillna("0", subset=[col])

        if type in ["int", 'long']:
            df = df.withColumn(col, df[col].cast(IntegerType())) #Rename and change to string type
            df = df.fillna(0, subset=[col])

        if type in ["float","double"]:
            df = df.withColumn(col, df[col].cast(FloatType())) #Rename and change to string type
            df = df.fillna(0, subset=[col])
    return df

#%%
df = TypeConvetion(data_set_schema, df)
# %%
# Data Prep function
def MLClassifierDFPrep(df,input_columns,dependent_var,treat_outliers=False,treat_neg_values=True):
    
    # change label (class variable) to string type to prep for reindexing
    # Pyspark is expecting a zero indexed integer for the label column. 
    # Just incase our data is not in that format... we will treat it by using the StringIndexer built in method
    renamed = df.withColumn("label_str", df[dependent_var].cast(StringType())) #Rename and change to string type
    indexer = StringIndexer(inputCol="label_str", outputCol="label") #Pyspark is expecting the this naming convention 
    indexed = indexer.fit(renamed).transform(renamed)

    # Convert all string type data in the input column list to numeric
    # Otherwise the Algorithm will not be able to process it
    numeric_inputs = []
    string_inputs = []
    for column in input_columns:
        if str(indexed.schema[column].dataType) == 'StringType':
            indexer = StringIndexer(inputCol=column, outputCol=column+"_num") 
            indexed = indexer.fit(indexed).transform(indexed)
            new_col_name = column+"_num"
            string_inputs.append(new_col_name)
        else:
            numeric_inputs.append(column)
    string_inputs.sort()
    numeric_inputs.sort() 
    print(numeric_inputs, string_inputs)    
    if treat_outliers == True:
        print("We are correcting for non normality now!")
        # empty dictionary d
        d = {}
        # Create a dictionary of quantiles
        for col in numeric_inputs: 
            d[col] = indexed.approxQuantile(col,[0.01,0.99],0.25) #if you want to make it go faster increase the last number
        #Now fill in the values
        for col in numeric_inputs:
            skew = indexed.agg(skewness(indexed[col])).collect() #check for skewness
            skew = skew[0][0]
            # This function will floor, cap and then log+1 (just in case there are 0 values)
            if skew > 1:
                indexed = indexed.withColumn(col, \
                log(when(df[col] < d[col][0],d[col][0])\
                .when(indexed[col] > d[col][1], d[col][1])\
                .otherwise(indexed[col] ) +1).alias(col))
                print(col+" has been treated for positive (right) skewness. (skew =)",skew,")")
            elif skew < -1:
                indexed = indexed.withColumn(col, \
                exp(when(df[col] < d[col][0],d[col][0])\
                .when(indexed[col] > d[col][1], d[col][1])\
                .otherwise(indexed[col] )).alias(col))
                print(col+" has been treated for negative (left) skewness. (skew =",skew,")")

            
    # Produce a warning if there are negative values in the dataframe that Naive Bayes cannot be used. 
    # Note: we only need to check the numeric input values since anything that is indexed won't have negative values
    minimums = df.select([min(c).alias(c) for c in df.columns if c in numeric_inputs]) # Calculate the mins for all columns in the df
    min_array = minimums.select(array(numeric_inputs).alias("mins")) # Create an array for all mins and select only the input cols
    df_minimum = min_array.select(array_min(min_array.mins)).collect() # Collect golobal min as Python object
    df_minimum = df_minimum[0][0] # Slice to get the number itself
    print(df_minimum)

    features_list = numeric_inputs + string_inputs
    assembler = VectorAssembler(inputCols=features_list,outputCol='features')
    output = assembler.transform(indexed).select('features','label')

#     final_data = output.select('features','label') #drop everything else
    
    # Now check for negative values and ask user if they want to correct that? 
    if (isinstance(df_minimum,(int,float)) and (df_minimum < 0)):
        print("WARNING: The Naive Bayes Classifier will not be able to process your dataframe as it contains negative values")
        print(" ")
        
    if treat_neg_values == True:
        print("You have opted to correct that by rescaling all your features to a range of 0 to 1")
        print(" ")
        print("We are rescaling you dataframe....")
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

        # Compute summary statistics and generate MinMaxScalerModel
        scalerModel = scaler.fit(output)

        # rescale each feature to range [min, max].
        scaled_data = scalerModel.transform(output)
        final_data = scaled_data.select('label','scaledFeatures')
        final_data = final_data.withColumnRenamed('scaledFeatures','features')
        print("Done!")

    else:
        print("You have opted not to correct that therefore you will not be able to use to Naive Bayes classifier")
        print("We will return the dataframe unscaled.")
        final_data = output
    
    return final_data
# %%
# col_list = ["USER_ID","CABIN_TYPE","USER_RESIDENCE"]

# input_columns = df.columns # Collect the column names as a list
# input_columns = input_columns[8:] # keep only relevant columns: from column 8 until the end
dependent_var = _settings.get('target_column')
all_column_list = [schema.get("name") for schema in data_set_schema]
col_list = list(set(all_column_list) - set([dependent_var]))
input_columns = col_list


#%%
final_data = MLClassifierDFPrep(df,input_columns,dependent_var)
final_data.limit(5).toPandas()
# %%
train,test = final_data.randomSplit([0.7,0.3])
# %%
def ClassTrainEval(classifier,features,classes,train,test):

    def FindMtype(classifier):
        # Intstantiate Model
        M = classifier
        # Learn what it is
        Mtype = type(M).__name__
        
        return Mtype
    
    Mtype = FindMtype(classifier)
    

    def IntanceFitModel(Mtype,classifier,classes,features,train):
        
        if Mtype == "OneVsRest":
            # instantiate the base classifier.
            lr = LogisticRegression()
            # instantiate the One Vs Rest Classifier.
            OVRclassifier = OneVsRest(classifier=lr)
#             fitModel = OVRclassifier.fit(train)
            # Add parameters of your choice here:
            paramGrid = ParamGridBuilder() \
                .addGrid(lr.regParam, [0.1, 0.01]) \
                .build()
            #Cross Validator requires the following parameters:
            crossval = CrossValidator(estimator=OVRclassifier,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=MulticlassClassificationEvaluator(),
                                      numFolds=4) # 3 is best practice
            # Run cross-validation, and choose the best set of parameters.
            fitModel = crossval.fit(train)
            return fitModel
        if Mtype == "MultilayerPerceptronClassifier":
            # specify layers for the neural network:
            # input layer of size features, two intermediate of features+1 and same size as features
            # and output of size number of classes
            # Note: crossvalidator cannot be used here
            features_count = len(features[0][0])
            
            # layers = [features_count, features_count+1, features_count, classes]
            layers = [features_count, 64, classes]
            MPC_classifier = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
            fitModel = MPC_classifier.fit(train)
            return fitModel
        if Mtype in("LinearSVC","GBTClassifier") and classes != 2: # These classifiers currently only accept binary classification
            print(Mtype," could not be used because PySpark currently only accepts binary classification data for this algorithm")
            return
        if Mtype in("LogisticRegression","NaiveBayes","RandomForestClassifier","GBTClassifier","LinearSVC","DecisionTreeClassifier"):
  
            # Add parameters of your choice here:
            if Mtype in("LogisticRegression"):
                paramGrid = (ParamGridBuilder() \
#                              .addGrid(classifier.regParam, [0.1, 0.01]) \
                             .addGrid(classifier.maxIter, [10, 15,20])
                             .build())
                
            # Add parameters of your choice here:
            if Mtype in("NaiveBayes"):
                paramGrid = (ParamGridBuilder() \
                             .addGrid(classifier.smoothing, [0.0, 0.2, 0.4, 0.6]) \
                             .build())
                
            # Add parameters of your choice here:
            if Mtype in("RandomForestClassifier"):
                paramGrid = (ParamGridBuilder() \
                               .addGrid(classifier.maxDepth, [2, 5, 10])
#                                .addGrid(classifier.maxBins, [5, 10, 20])
#                                .addGrid(classifier.numTrees, [5, 20, 50])
                             .build())
                
            # Add parameters of your choice here:
            if Mtype in("GBTClassifier"):
                paramGrid = (ParamGridBuilder() \
#                              .addGrid(classifier.maxDepth, [2, 5, 10, 20, 30]) \
#                              .addGrid(classifier.maxBins, [10, 20, 40, 80, 100]) \
                             .addGrid(classifier.maxIter, [10, 15,50,100])
                             .build())
                
            # Add parameters of your choice here:
            if Mtype in("LinearSVC"):
                paramGrid = (ParamGridBuilder() \
                             .addGrid(classifier.maxIter, [10, 15]) \
                             .addGrid(classifier.regParam, [0.1, 0.01]) \
                             .build())
            
            # Add parameters of your choice here:
            if Mtype in("DecisionTreeClassifier"):
                paramGrid = (ParamGridBuilder() \
#                              .addGrid(classifier.maxDepth, [2, 5, 10, 20, 30]) \
                             .addGrid(classifier.maxBins, [10, 20, 40, 80, 100]) \
                             .build())
            
            #Cross Validator requires all of the following parameters:
            crossval = CrossValidator(estimator=classifier,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=MulticlassClassificationEvaluator(),
                                      numFolds=4) # 3 + is best practice
            # Fit Model: Run cross-validation, and choose the best set of parameters.
            fitModel = crossval.fit(train)
            return fitModel
    
    fitModel = IntanceFitModel(Mtype,classifier,classes,features,train)
    
    # Print feature selection metrics
    if fitModel is not None:
        
        if Mtype in("OneVsRest"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print('\033[1m' + Mtype + '\033[0m')
            # Extract list of binary models
            models = BestModel.models
            for model in models:
                print('\033[1m' + 'Intercept: '+ '\033[0m',model.intercept,'\033[1m' + '\nCoefficients:'+ '\033[0m',model.coefficients)

        if Mtype == "MultilayerPerceptronClassifier":
            print("")
            print('\033[1m' + Mtype," Weights"+ '\033[0m')
            print('\033[1m' + "Model Weights: "+ '\033[0m',fitModel.weights.size)
            print("")

        if Mtype in("DecisionTreeClassifier", "GBTClassifier","RandomForestClassifier"):
            # FEATURE IMPORTANCES
            # Estimate of the importance of each feature.
            # Each feature’s importance is the average of its importance across all trees 
            # in the ensemble The importance vector is normalized to sum to 1. 
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print('\033[1m' + Mtype," Feature Importances"+ '\033[0m')
            print("(Scores add up to 1)")
            print("Lowest score is the least important")
            print(" ")
            featureImportances = BestModel.featureImportances.toArray()
            print(featureImportances)
            
            if Mtype in("DecisionTreeClassifier"):
                global DT_featureImportances
                DT_featureImportances = BestModel.featureImportances.toArray()
                global DT_BestModel
                DT_BestModel = BestModel
            if Mtype in("GBTClassifier"):
                global GBT_featureImportances
                GBT_featureImportances = BestModel.featureImportances.toArray()
                global GBT_BestModel
                GBT_BestModel = BestModel
            if Mtype in("RandomForestClassifier"):
                global RF_featureImportances
                RF_featureImportances = BestModel.featureImportances.toArray()
                global RF_BestModel
                RF_BestModel = BestModel

        if Mtype in("LogisticRegression"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print('\033[1m' + Mtype," Coefficient Matrix"+ '\033[0m')
            print("You should compares these relative to eachother")
            print("Coefficients: \n" + str(BestModel.coefficientMatrix))
            print("Intercept: " + str(BestModel.interceptVector))
            global LR_coefficients
            LR_coefficients = BestModel.coefficientMatrix.toArray()
            global LR_BestModel
            LR_BestModel = BestModel

        if Mtype in("LinearSVC"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print('\033[1m' + Mtype," Coefficients"+ '\033[0m')
            print("You should compares these relative to eachother")
            print("Coefficients: \n" + str(BestModel.coefficients))
            global LSVC_coefficients
            LSVC_coefficients = BestModel.coefficients.toArray()
            global LSVC_BestModel
            LSVC_BestModel = BestModel
        
   
    # Set the column names to match the external results dataframe that we will join with later:
    columns = ['Classifier', 'Result']
    
    if Mtype in("LinearSVC","GBTClassifier") and classes != 2:
        Mtype = [Mtype] # make this a list
        score = ["N/A"]
        result = spark.createDataFrame(zip(Mtype,score), schema=columns)
    else:
        predictions = fitModel.transform(test)
        # print(predictions.limit(4).toPandas())
        MC_evaluator = MulticlassClassificationEvaluator(metricName="accuracy") # redictionCol="prediction",
        accuracy = (MC_evaluator.evaluate(predictions))*100
        Mtype = [Mtype] # make this a string
        score = [str(accuracy)] #make this a string and convert to a list
        result = spark.createDataFrame(zip(Mtype,score), schema=columns)
        result = result.withColumn('Result',result.Result.substr(0, 5))
        
    return result
    #Also returns the fit model important scores or p values
# %%
# Run!
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.sql import functions
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Comment out Naive Bayes if your data still contains negative values
classifier_support = {
    "LogisticRegression":LogisticRegression,
    "LinearSVC":LinearSVC,
    "NaiveBayes":NaiveBayes,
    "RandomForestClassifier":RandomForestClassifier,
    "GBTClassifier":GBTClassifier,
    "DecisionTreeClassifier":DecisionTreeClassifier,
    "MultilayerPerceptronClassifier":MultilayerPerceptronClassifier




}
#%%
classifiers = [classifier_support.get(settings.get("name"))() for settings in _settings.get("algorithm_list") if not isinstance(settings.get("name"),type(None))]
#%%
# classifiers = [
#                 LogisticRegression(),
#                 # OneVsRest(),
#                 LinearSVC(),
#                 NaiveBayes(),
#                RandomForestClassifier(),
#                GBTClassifier(),
#                DecisionTreeClassifier(),
#              MultilayerPerceptronClassifier()
#               ] 

train,test = final_data.randomSplit([0.7,0.3])
features = final_data.select(['features']).collect()
# Learn how many classes there are in order to specify evaluation type based on binary or multi and turn the df into an object
class_count = final_data.select(countDistinct("label")).collect()
classes = class_count[0][0]

#set up your results table
columns = ['Classifier', 'Result']
vals = [("Place Holder","N/A")]
results = spark.createDataFrame(vals, columns)

for classifier in classifiers:
    new_result = ClassTrainEval(classifier,features,classes,train,test)
    results = results.union(new_result)
results = results.where("Classifier!='Place Holder'")

# %%
results.show(100,False)

# %%
