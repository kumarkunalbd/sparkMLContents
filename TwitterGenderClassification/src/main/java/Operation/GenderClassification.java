package Operation;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.tree.DecisionTreeParams;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import breeze.linalg.where;


public class GenderClassification {
	
	private static UDF1 hexToLong = new UDF1<String, String>() {
		public String call(final String str) throws Exception {
			try {
				String longStrvalue = String.valueOf(java.lang.Long.parseLong(str.trim(), 16));
				return longStrvalue;
				/*Long longValue = java.lang.Long.parseLong(str.trim(), 16);
				return longValue;*/
				
			} catch (Exception e) {
				// TODO: handle exception
				return "-1";
				//return (long) 0;
			}
			
		}
	};
	
	

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
				Logger.getLogger("org").setLevel(Level.ERROR);
				Logger.getLogger("akka").setLevel(Level.ERROR);

				SparkSession sparkSession = SparkSession.builder()  //SparkSession  
						.appName("SparkML") 
						.master("local[*]") 
						.getOrCreate(); //

				// Read the file as a training dataset
				String pathTrain = "data/gender-classifier.csv";	
				//Dataset<Row> mainDataSet = sparkSession.read().format("csv").option("header","true").option("ignoreLeadingWhiteSpace",false).option("ignoreTrailingWhiteSpace",false).load(pathTrain);
				Dataset<Row> mainDataSet = sparkSession.read().option("header", true).option("inferschema", true).option("mode", "DROPMALFORMED").csv(pathTrain);
				mainDataSet.printSchema();
				mainDataSet.describe().show();
				System.out.println("mainDataSet Row Count  ::"+mainDataSet.count());
				
				/*********** Filtering records for gender female,male and brand *********************/
				String filteringColumn = "gender";
				mainDataSet = mainDataSet.filter(col(filteringColumn).equalTo("female").or(col(filteringColumn).equalTo("male")));
				System.out.println("mainDataSet Row Count after filetring ::"+mainDataSet.count());
				
				//Removing Rows with missing values
				System.out.println("********************Removing records with missing values**********************");
				Dataset<Row> framingDataSet = mainDataSet.select(col("gender"),col("description"),col("link_color"),col("name"),col("sidebar_color"),col("text"));
				framingDataSet = framingDataSet.na().drop();
				System.out.println("framingDataSet Row Count after null drop ::"+framingDataSet.count());
				// drop the columns which are non relevant
				/*Dataset<Row> selectedColumnDataSet = mainDataSet.select(mainDataSet.col("gender"));
				System.out.println("Dataset<Row> selectedColumnDataSet: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());
				String[] dropCol = {"_golden","_unit_state","_trusted_judgments","_last_judgment_at","gender:confidence","profile_yn:confidence","created","fav_number","gender_gold","profileimage","profile_yn_gold","tweet_coord","tweet_id","tweet_created","tweet_location",
						"user_timezone"};
				for(String column: dropCol) {
					selectedColumnDataSet = selectedColumnDataSet.drop(column);
				}
				System.out.println("After drop Dataset<Row> selectedColumnDataSet: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());

				selectedColumnDataSet.show();
				selectedColumnDataSet.printSchema();
				selectedColumnDataSet.describe().show();*/
				
				
				
				// drop the rows containing any null or NaN values
				/*selectedColumnDataSet = selectedColumnDataSet.na().drop();
				System.out.println("Droping Rows containing null in csv: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());
				selectedColumnDataSet.show();*/
				
				//Label - HextoLong - process to convert it to a double variable
				Dataset<Row> selectedColumnDataSet = framingDataSet;
				selectedColumnDataSet.printSchema();
				selectedColumnDataSet.show();
				sparkSession.udf().register("toHexLong", hexToLong, DataTypes.StringType);
				selectedColumnDataSet = selectedColumnDataSet.withColumn("link_color_long", callUDF("toHexLong", selectedColumnDataSet.col("link_color"))).drop("link_color");
				selectedColumnDataSet = selectedColumnDataSet.withColumn("sidebar_color_long", callUDF("toHexLong", selectedColumnDataSet.col("sidebar_color"))).drop("sidebar_color");
				System.out.println("After HexToLong csv: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());
				System.out.println("********************selectedColumnDataSet after changing values of linkcolor and sidecolor to long**********************");
				selectedColumnDataSet.printSchema();
				selectedColumnDataSet.show();
				System.out.println("********************selectedColumnDataSet femoving garbage link_color and sidebar color**********************");
				selectedColumnDataSet = selectedColumnDataSet.filter(col("link_color_long").notEqual("0").and(col("link_color_long").notEqual("-1")));
				System.out.println("selectedColumnDataSet femoving garbage link_color: " +selectedColumnDataSet.count());
				selectedColumnDataSet = selectedColumnDataSet.filter(col("sidebar_color_long").notEqual("0").and(col("sidebar_color_long").notEqual("-1")));
				System.out.println("selectedColumnDataSet femoving garbage sidebar_color_long: " +selectedColumnDataSet.count());
				selectedColumnDataSet.printSchema();
				selectedColumnDataSet.show();
				/*System.out.println("********************selectedColumnDataSet after casting link color and side bar color**********************");
				selectedColumnDataSet = selectedColumnDataSet.select(col("gender"),col("description"),col("link_color_long").cast(DataTypes.LongType),col("name"),col("sidebar_color_long").cast(DataTypes.LongType),col("text"));
				selectedColumnDataSet.printSchema();
				selectedColumnDataSet.show();	*/			
								 				
				// Split data into Training and testing
				Dataset<Row>[] dataSplit = selectedColumnDataSet.randomSplit(new double[] { 0.7, 0.3 });
				Dataset<Row> traindata = dataSplit[0];
				Dataset<Row> testdata = dataSplit[1];
				
				System.out.println("TrainData Row Count  ::"+traindata.count());
				
				/*String[] inputCols = {"description","text"};
				VectorAssembler assembler = new
						VectorAssembler().setInputCols(new String[]{"description","text"}).setOutputCol("features1");
				Dataset<Row> combinedDataSet = assembler.transform(selectedColumnDataSet).select("gender","features1");
				combinedDataSet.show();*/
				
				String selectedColumn = "description";
				String selectedColumn2 = "text";
				String selectedColumn3 = "name";
				String selectedColumn4 = "link_color_long";
				String selectedColumn5 = "sidebar_color_long";
				
				traindata = traindata.select(traindata.col("gender"), traindata.col(selectedColumn), traindata.col(selectedColumn3),traindata.col(selectedColumn2),traindata.col(selectedColumn4),traindata.col(selectedColumn5));
				traindata.show();
				
				testdata = testdata.select(testdata.col("gender"), testdata.col(selectedColumn),traindata.col(selectedColumn3),traindata.col(selectedColumn2),traindata.col(selectedColumn4),traindata.col(selectedColumn5));
				testdata.show();
				
				// Configure an ML pipeline, which consists of multiple stages: indexer, tokenizer, hashingTF, idf, lr/rf etc 
				// and labelindexer.		
				//Relabel the target variable
				StringIndexerModel labelindexer = new StringIndexer()
						.setInputCol("gender")
						.setOutputCol("label").setHandleInvalid("keep").fit(traindata);
				
				// Tokenize the input text
				Tokenizer tokenizer = new Tokenizer()
						.setInputCol(selectedColumn)
						.setOutputCol("tokenizerwords1");
				//Dataset<Row> df1Tokenizer = tokenizer.transform(traindata);
				
				// Remove the stop words
				StopWordsRemover remover = new StopWordsRemover()
						.setInputCol(tokenizer.getOutputCol())
						.setOutputCol("filtered1");
				//Dataset<Row> df1remover = remover.transform(df1Tokenizer);
				
				
				// Create the Term Frequency Matrix
				HashingTF hashingTF = new HashingTF()
						.setNumFeatures(1000)
						.setInputCol(remover.getOutputCol())
						.setOutputCol("numFeatures1");
				//Dataset<Row> df1Hashinngtf = hashingTF.transform(df1remover);
				
				

				// Calculate the Inverse Document Frequency 
				IDF idf = new IDF()
						.setInputCol(hashingTF.getOutputCol())
						.setOutputCol("features1");
				
				
				
				/*Pipeline pipeline = new Pipeline().setStages(new PipelineStage [] {tokenizer, remover,hashingTF, idf});
				PipelineModel model = pipeline.fit(traindata);
				Dataset<Row> df1IdfSet = model.transform(traindata);
				df1IdfSet.show();
				System.out.println("df1IdfSet Row Count  ::"+df1IdfSet.count());
				*/
				Tokenizer tokenizer2 = new Tokenizer()
						.setInputCol(selectedColumn2)
						.setOutputCol("tokenizerwords2");
				/*tokenizer.setInputCol(selectedColumn2).setOutputCol("tokenizerwords3");
				Dataset<Row> df4 = tokenizer.transform(df1IdfSet);    */
				// Remove the stop words
				StopWordsRemover remover2 = new StopWordsRemover()
						.setInputCol(tokenizer2.getOutputCol())
						.setOutputCol("filtered2");
				
				//remover.setInputCol("tokenizerwords3").setOutputCol("filtered2");
				//Dataset<Row> df5 = tokenizer.transform(df4); 
				
				// Create the Term Frequency Matrix
				HashingTF hashingTF2 = new HashingTF()
						.setNumFeatures(1000)
						.setInputCol(remover2.getOutputCol())
						.setOutputCol("numFeatures2");
				//hashingTF.setInputCol("filtered2").setOutputCol("numFeatures2");
				//Dataset<Row> df6 = tokenizer.transform(df5); 
				
				// Calculate the Inverse Document Frequency 
				IDF idf2 = new IDF()
						.setInputCol(hashingTF2.getOutputCol())
						.setOutputCol("features2");
				//idf.setInputCol("numFeatures2").setOutputCol("features2");
				//Dataset<Row> df7 = tokenizer.transform(df6);
				
				/* Idf for 3rd column */
				
				Tokenizer tokenizer3 = new Tokenizer()
						.setInputCol(selectedColumn3)
						.setOutputCol("tokenizerwords3");
				
				StopWordsRemover remover3 = new StopWordsRemover()
						.setInputCol(tokenizer3.getOutputCol())
						.setOutputCol("filtered3");
				
				HashingTF hashingTF3 = new HashingTF()
						.setNumFeatures(1000)
						.setInputCol(remover3.getOutputCol())
						.setOutputCol("numFeatures3");
				
				IDF idf3 = new IDF()
						.setInputCol(hashingTF3.getOutputCol())
						.setOutputCol("features3");
				
				/* String indexing and tokenizing for link_color and side color*/
				Tokenizer tokenizer4 = new Tokenizer()
						.setInputCol(selectedColumn4)
						.setOutputCol("tokenizerwords4");
				
				HashingTF hashingTF4 = new HashingTF()
						.setNumFeatures(1000)
						.setInputCol(tokenizer4.getOutputCol())
						.setOutputCol("numFeatures4");
				
				IDF idf4 = new IDF()
						.setInputCol(hashingTF4.getOutputCol())
						.setOutputCol("features4");
				
				Tokenizer tokenizer5 = new Tokenizer()
						.setInputCol(selectedColumn5)
						.setOutputCol("tokenizerwords5");
				
				HashingTF hashingTF5 = new HashingTF()
						.setNumFeatures(1000)
						.setInputCol(tokenizer5.getOutputCol())
						.setOutputCol("numFeatures5");
				
				IDF idf5 = new IDF()
						.setInputCol(hashingTF5.getOutputCol())
						.setOutputCol("features5");
				
				
				
				 //Assembling the features in the dataFrame as Dense Vector
		        VectorAssembler assemblerColors = new VectorAssembler()
		                .setInputCols(new String[]{"link_color_label","sidebar_color_label"})
		                .setOutputCol("featuresColor");
		        
		        
				
				/*Pipeline pipeline2 = new Pipeline().setStages(new PipelineStage [] {tokenizer, remover,hashingTF, idf});
				PipelineModel model2 = pipeline2.fit(traindata);
				Dataset<Row> df1IdfSet2 = model2.transform(traindata);*/
				
				//df1IdfSet2.show();
				
				/*System.out.println("After second tokenizwer,and hashingTF:");
				
				df1IdfSet2 = df1IdfSet2.drop("gender").drop("description").drop("text");
				df1IdfSet2.show();
				System.out.println("df1IdfSet2 Row Count  ::"+df1IdfSet2.count());
				System.out.println("Perform Inner Join Operation");
				Dataset<Row> df1IdfSetJoined = df1IdfSet.join(df1IdfSet2, "_unit_id");
				df1IdfSetJoined.printSchema();
				df1IdfSetJoined.show();
				System.out.println("df1IdfSetJoined Row Count  ::"+df1IdfSetJoined.count());*/
				
				// Now Assemble Vectors
				VectorAssembler assembler = new VectorAssembler()
				        .setInputCols(new String[]{"features1", "features2", "features3","features4"})
				        .setOutputCol("features");
				
				//Scaling the features between 0-1
				MaxAbsScaler scaler = new MaxAbsScaler() //Performing MaxAbsScaler() Transformation
						.setInputCol("features")
						.setOutputCol("scaledFeatures");
				

				//*********************************************Normalizing the Vector*********************************************************************

				//Normalizing the vector. Converts vector to a unit vector
				Normalizer normalizer = new Normalizer() //Performing Normalizer() Transformation
						.setInputCol("scaledFeatures")
						.setOutputCol("normFeatures")
						.setP(2.0);

				//Dataset<Row> assembledFeatures = assembler.transform(df1IdfSetJoined);
				
				//assembledFeatures.show();
				
				/*String idfColumnn1 = idf.getOutputCol();
				String idfColumnn2 = idf2.getOutputCol();*/

				
				
				/*VectorAssembler assembler = new
						VectorAssembler().setInputCols(new String[]{idfColumnn1,idfColumnn2}).setOutputCol("features");*/
				
				// Set up the Random Forest Model
				RandomForestClassifier rf = new RandomForestClassifier();
				rf.setMaxDepth(16);
				rf.setMinInfoGain(0.0);
				

				//Set up Decision Tree
				DecisionTreeClassifier dt = new DecisionTreeClassifier();
				dt.setMaxDepth(16);
				dt.setMinInfoGain(0.0);
				

				// Convert indexed labels back to original labels once prediction is available	
				IndexToString labelConverter = new IndexToString()
						.setInputCol("prediction")
						.setOutputCol("predictedLabel").setLabels(labelindexer.labels());
				
				// Create and Run Random Forest Pipeline
				Pipeline pipelineRF = new Pipeline()
						.setStages(new PipelineStage[] {labelindexer,tokenizer, remover,hashingTF,idf,tokenizer2,remover2,hashingTF2,idf2,tokenizer3,remover3,hashingTF3,idf3,tokenizer4,hashingTF4,idf4,tokenizer5,hashingTF5,idf5,assembler,scaler,normalizer,rf,labelConverter});	
				// Fit the pipeline to training documents.
				PipelineModel modelRF = pipelineRF.fit(traindata);		
				// Make predictions on test documents.
				Dataset<Row> predictionsRF = modelRF.transform(testdata);
				System.out.println("Predictions from Random Forest Model are:");
				predictionsRF.show(10);

				// Create and Run Decision Tree Pipeline
				Pipeline pipelineDT = new Pipeline()
						.setStages(new PipelineStage[] {labelindexer,tokenizer, remover,hashingTF, idf,tokenizer2,remover2, hashingTF2,idf2,tokenizer3,remover3,hashingTF3,idf3,tokenizer4,hashingTF4,idf4,assembler,scaler,normalizer,dt,labelConverter});	
				// Fit the pipeline to training documents.
				PipelineModel modelDT = pipelineDT.fit(traindata);		
				// Make predictions on test documents.
				Dataset<Row> predictionsDT = modelDT.transform(testdata);
				System.out.println("Predictions from Decision Tree Model are:");
				predictionsDT.show(10);	
				
				/*************************Model Evaluation*********************/
				
				//View confusion matrix
				System.out.println("Confusion Matrix from Decision Tree :");
				predictionsDT.groupBy(col("label"),col("predictedLabel")).count().show();
				
				// Select (prediction, true label) and compute test error.
				MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
						.setLabelCol("label")
						.setPredictionCol("prediction")
						.setMetricName("accuracy");	

				//Evaluate Random Forest
				double accuracyRF = evaluator.evaluate(predictionsRF);
				System.out.println("Test Error for Random Forest = " + (1.0 - accuracyRF));
				System.out.println("Accuracy for Random Forest= " + Math.round(accuracyRF * 100) + " %");
				

				//Evaluate Decision Tree
				double accuracyDT = evaluator.evaluate(predictionsDT);
				System.out.println("Test Error for Decision Tree = " + (1.0 - accuracyDT));	
				System.out.println("Accuracy for Decision Tree = " + Math.round(accuracyDT * 100) + " %");
				
				MulticlassClassificationEvaluator evaluatorPrecision = new MulticlassClassificationEvaluator()
						.setLabelCol("label")
						.setPredictionCol("prediction")
						.setMetricName("weightedPrecision");
				//Evaluate Random Forest
				double precisionRF = evaluatorPrecision.evaluate(predictionsRF);
				System.out.println("Precision for Random Forest= " + precisionRF);
				
				//Evaluate Decision Tree
				double precisionDT = evaluatorPrecision.evaluate(predictionsDT);
				System.out.println("Precision for Random Forest= " + precisionDT);
				
				MulticlassClassificationEvaluator evaluatorRecall = new MulticlassClassificationEvaluator()
						.setLabelCol("label")
						.setPredictionCol("prediction")
						.setMetricName("weightedRecall");
				//Evaluate Random Forest
				double recallRF = evaluatorRecall.evaluate(predictionsRF);
				System.out.println("Recall for Random Forest= " + recallRF);
				
				//Evaluate Decision Tree
				double recallDT = evaluatorRecall.evaluate(predictionsDT);
				System.out.println("Recall for Random Forest= " + recallDT);
				
				double fscoreRF = 2*precisionRF*recallRF/(precisionRF+recallRF);
				System.out.println("fScroe for Random Forest= " + fscoreRF);
				
				double fscoreDT = 2*precisionDT*recallDT/(precisionDT+recallDT);
				System.out.println("fScroe for Random Forest= " + fscoreDT);
				
	}

}
