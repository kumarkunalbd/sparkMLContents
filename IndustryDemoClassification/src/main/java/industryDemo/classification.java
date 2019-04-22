package industryDemo;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
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


public class classification {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		SparkSession sparkSession = SparkSession.builder()  //SparkSession  
				.appName("SparkML") 
				.master("local[*]") 
				.getOrCreate(); //

		// Read the file as a training dataset
		String pathTrain = "data/train1.csv";	
		Dataset<Row> trainset = sparkSession.read().format("csv").option("header","true").load(pathTrain);
		//Isolate the relevant columns
		Dataset<Row> traindata = trainset.select(trainset.col("Descript"), trainset.col("Category")); 
		traindata.show();

		// Read the file as a test dataset
		String pathTest = "data/test1.csv";
		Dataset<Row> testset = sparkSession.read().format("csv").option("header","true").load(pathTest);
		//Isolate the relevant columns
		Dataset<Row> testdata = testset.select(testset.col("Descript"), testset.col("Category")); 
		testdata.show();


		// Configure an ML pipeline, which consists of multiple stages: indexer, tokenizer, hashingTF, idf, lr/rf etc 
		// and labelindexer.		
		//Relabel the target variable
		StringIndexerModel labelindexer = new StringIndexer()
				.setInputCol("Category")
				.setOutputCol("label").fit(traindata);

		// Tokenize the input text
		Tokenizer tokenizer = new Tokenizer()
				.setInputCol("Descript")
				.setOutputCol("words");

		// Remove the stop words
		StopWordsRemover remover = new StopWordsRemover()
				.setInputCol(tokenizer.getOutputCol())
				.setOutputCol("filtered");		

		// Create the Term Frequency Matrix
		HashingTF hashingTF = new HashingTF()
				.setNumFeatures(1000)
				.setInputCol(remover.getOutputCol())
				.setOutputCol("numFeatures");

		// Calculate the Inverse Document Frequency 
		IDF idf = new IDF()
				.setInputCol(hashingTF.getOutputCol())
				.setOutputCol("features");

		// Set up the Random Forest Model
		RandomForestClassifier rf = new RandomForestClassifier();

		//Set up Decision Tree
		DecisionTreeClassifier dt = new DecisionTreeClassifier();

		// Convert indexed labels back to original labels once prediction is available	
		IndexToString labelConverter = new IndexToString()
				.setInputCol("prediction")
				.setOutputCol("predictedLabel").setLabels(labelindexer.labels());

		// Create and Run Random Forest Pipeline
		Pipeline pipelineRF = new Pipeline()
				.setStages(new PipelineStage[] {labelindexer, tokenizer, remover, hashingTF, idf, rf,labelConverter});	
		// Fit the pipeline to training documents.
		PipelineModel modelRF = pipelineRF.fit(traindata);		
		// Make predictions on test documents.
		Dataset<Row> predictionsRF = modelRF.transform(testdata);
		System.out.println("Predictions from Random Forest Model are:");
		predictionsRF.show(10);

		// Create and Run Decision Tree Pipeline
		Pipeline pipelineDT = new Pipeline()
				.setStages(new PipelineStage[] {labelindexer, tokenizer, remover, hashingTF, idf, dt,labelConverter});	
		// Fit the pipeline to training documents.
		PipelineModel modelDT = pipelineDT.fit(traindata);		
		// Make predictions on test documents.
		Dataset<Row> predictionsDT = modelDT.transform(testdata);
		System.out.println("Predictions from Decision Tree Model are:");
		predictionsDT.show(10);		

		// Select (prediction, true label) and compute test error.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("accuracy");		

		//Evaluate Random Forest
		double accuracyRF = evaluator.evaluate(predictionsRF);
		System.out.println("Test Error for Random Forest = " + (1.0 - accuracyRF));

		//Evaluate Decision Tree
		double accuracyDT = evaluator.evaluate(predictionsDT);
		System.out.println("Test Error for Decision Tree = " + (1.0 - accuracyDT));
	}

}
