 //package com.data.assignment;
 package com.data.rf.assignment;


import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.*; //.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class UserGenderRF_Assignment {

	public static void main(String[] args) {
		
		//Printing only error logger messages
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		//SparkSession construction
		SparkSession sparkSession = SparkSession.builder().appName("SparkML").master("local[*]").getOrCreate();

		//Load the Dataset with gender related CSV file from respective location 
		//Reading Data from a CSV file //Inferring Schema and Setting Header as True
		Dataset<Row> genderDf = sparkSession.read().option("header", true).option("inferSchema",true).csv("data/gender-classifier-DFE-791531.csv");
                
		//Displaying Samples
		genderDf.show(); 
		
		//Printing Schema
        genderDf.printSchema(); 
        
        //Statistically Summarizing about the data
        genderDf.describe().show(); 
        
      //****************************************** Handling missing values ******************************************************************
        
        //Casting few attributes from String to Double
        Dataset<Row> dataframemissing = genderDf.select(col("gender"),col("description"),col("profile_yn"),col("text"),col("gender_gold"),col("gender:confidence").cast("Float"),col("link_color"),col("fav_number"));
        		
        System.out.println("*************************Casting columns*****************************");
        
        //Displaying samples 
        dataframemissing.show(); 
        
        //Printing new Schema
        dataframemissing.printSchema(); 
               
		//Removing Rows with missing values
		System.out.println("********************Removing records with missing values**********************");
		
		//Dataframe.na.drop removes any row with a NULL value
		Dataset<Row> dataframewithoutnull = dataframemissing.na().drop(); 
		
		Dataset<Row> dataframewithoutnull1=dataframewithoutnull.drop("brand");
		
		//Describing DataFrame
		dataframewithoutnull1.describe().show(); 
  
		//****************************************Data Splitting with specific rows ************************************************************

		System.out.println("******************************Trainig and Testing Data build************************");
		
		//Load the Dataset with the gender, text and description columns data
		Dataset<Row> dataframespecificrows= dataframewithoutnull1.select(col("gender"),col("gender:confidence"),col("text"),col("link_color"),col("profile_yn"),col("fav_number"));
		
		//Dataframe.na.drop removes any row with a NULL value
		Dataset<Row> dataframerows = dataframespecificrows.na().drop(); 
		Dataset<Row> dataframerows1 = dataframespecificrows.drop("brand"); 
		System.out.println("******************************We Are here************************");

		
		//Describing DataFrame
		dataframerows1.describe().show(); 
		
		//Training Data
        Dataset<Row>[] splits = dataframerows1.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> traindata = splits[0];	
		System.out.println("******************************Is this the one printing************************");

		traindata.show();
		
        traindata.count();
        
        //Testing Data
        Dataset<Row> testdata = splits[1];			
        testdata.show();
        testdata.count();
 
		//Configure an ML pipeline, which consists of multiple stages: indexer, tokenizer, hashingTF, idf, lr/rf etc and labelindexer.		
		//Relabel the target variable
		StringIndexerModel labelindexer = new StringIndexer().setInputCol("gender").setOutputCol("label").fit(traindata);

		//Tokenize the input text
		Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");

		//Remove the stop words
		StopWordsRemover remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol()).setOutputCol("filtered");		

		//Create the Term Frequency Matrix
		HashingTF hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(remover.getOutputCol()).setOutputCol("numFeatures");

		//Calculate the Inverse Document Frequency 
		IDF idf = new IDF().setInputCol(hashingTF.getOutputCol()).setOutputCol("features");

		//Set up the Random Forest Model
		RandomForestClassifier rf = new RandomForestClassifier();

		//Set up Decision Tree
		DecisionTreeClassifier dt = new DecisionTreeClassifier();
		
		//Convert indexed labels back to original labels once prediction is available	
		IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelindexer.labels());

		//Create and Run Random Forest Pipeline
		Pipeline pipelineRF = new Pipeline().setStages(new PipelineStage[] {labelindexer, tokenizer, remover, hashingTF, idf, rf,labelConverter});	
		
		//Fit the pipeline to training documents.
		PipelineModel modelRF = pipelineRF.fit(traindata);	
		
		//Make predictions on test documents.
		Dataset<Row> predictionsRF = modelRF.transform(testdata);
		System.out.println("Predictions from Random Forest Model are:");
		predictionsRF.show(10);

		//Create and Run Decision Tree Pipeline
		Pipeline pipelineDT = new Pipeline().setStages(new PipelineStage[] {labelindexer, tokenizer, remover, hashingTF, idf, dt,labelConverter});	
		
		//Fit the pipeline to training documents.
		PipelineModel modelDT = pipelineDT.fit(traindata);		
		
		//Make predictions on test documents.
		Dataset<Row> predictionsDT = modelDT.transform(testdata);
		System.out.println("Predictions from Decision Tree Model are:  ");
		predictionsDT.show(10);		

		/*************************Model Evaluation*********************/
		//View confusion matrix
		System.out.println("Confusion Matrix from Decision Tree :");
		predictionsDT.groupBy(col("label"),col("predictedLabel")).count().show();
		
		//Select (prediction, true label) and compute test error.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy");		

		//Accuracy computation Evaluate Random Forest
		double accuracyRF = evaluator.evaluate(predictionsRF);
		System.out.println("Accuracy for Random Forest = " + Math.round(accuracyRF * 100) + " %");
		System.out.println("Test Error for Random Forest = " + (1.0 - accuracyRF));

		//Accuracy computation Evaluate Decision Tree
		double accuracyDT = evaluator.evaluate(predictionsDT);
		System.out.println("Accuracy for Decision Tree = " + Math.round(accuracyDT * 100) + " %");
		System.out.println("Test Error for Decision Tree = " + (1.0 - accuracyDT));		
	}
}
