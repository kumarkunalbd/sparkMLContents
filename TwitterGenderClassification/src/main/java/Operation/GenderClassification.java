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
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;


public class GenderClassification {
	
	private static UDF1 hexToLong = new UDF1<String, String>() {
		public String call(final String str) throws Exception {
			try {
				String longStrvalue = String.valueOf(java.lang.Long.parseLong(str.trim(), 16));
				return longStrvalue;
			} catch (Exception e) {
				// TODO: handle exception
				return "garbageHexValue";
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
				Dataset<Row> mainDataSet = sparkSession.read().format("csv").option("header","true").option("ignoreLeadingWhiteSpace",false).option("ignoreTrailingWhiteSpace",false).load(pathTrain);
				mainDataSet.printSchema();
				mainDataSet.describe().show();
				
				// drop the columns which are non relevant
				Dataset<Row> selectedColumnDataSet = mainDataSet;
				System.out.println("Dataset<Row> selectedColumnDataSet: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());
				String[] dropCol = {"_golden","_unit_state","_trusted_judgments","_last_judgment_at","gender:confidence","profile_yn:confidence","created","fav_number","gender_gold","profileimage","profile_yn_gold","tweet_coord","tweet_id","tweet_created","tweet_location",
						"user_timezone"};
				for(String column: dropCol) {
					selectedColumnDataSet = selectedColumnDataSet.drop(column);
				}
				System.out.println("After drop Dataset<Row> selectedColumnDataSet: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());

				selectedColumnDataSet.show();
				selectedColumnDataSet.printSchema();
				selectedColumnDataSet.describe().show();
				
				// Filtering the null values
				/*Dataset<Row> containingNullsGender =  selectedColumnDataSet.where(selectedColumnDataSet.col("gender").isNull());
				containingNullsGender.show();
				System.out.println("Dataset<Row> containingNullsGender: "+containingNullsGender.count());*/
				
				/*Dataset<Row> containingNullsGenderConf =  selectedColumnDataSet.where(selectedColumnDataSet.col("gender:confidence").isNull());
				containingNullsGenderConf.show();
				System.out.println("Dataset<Row> containingNullsGenderConf: "+containingNullsGenderConf.count());*/
				
				/*Dataset<Row> containingNullsprofile_yn =  selectedColumnDataSet.where(selectedColumnDataSet.col("profile_yn").isNull());
				containingNullsprofile_yn.show();
				System.out.println("Dataset<Row> containingNullsprofile_yn: "+containingNullsprofile_yn.count());*/
				
				/*Dataset<Row> containingNullsprofile_ynConf =  selectedColumnDataSet.where(selectedColumnDataSet.col("profile_yn:confidence").isNull());
				containingNullsprofile_ynConf.show();
				System.out.println("Dataset<Row> containingNullsprofile_ynConf: "+containingNullsprofile_ynConf.count());*/
				
				/*Dataset<Row> containingNullsdescription =  selectedColumnDataSet.where(selectedColumnDataSet.col("description").isNull());
				containingNullsdescription.show();
				System.out.println("Dataset<Row> containingNullsdescription: "+containingNullsdescription.count());*/
				
				/*Dataset<Row> containingNullsFavnum =  selectedColumnDataSet.where(selectedColumnDataSet.col("fav_number").isNull());
				containingNullsFavnum.show();
				System.out.println("Dataset<Row> containingNullsFavnum: "+containingNullsFavnum.count());*/
				
				/*Dataset<Row> containingNullsLinkColor =  selectedColumnDataSet.where(selectedColumnDataSet.col("link_color").isNull());
				containingNullsLinkColor.show();
				System.out.println("Dataset<Row> containingNullsLinkColor: "+containingNullsLinkColor.count());*/
				
				/*Dataset<Row> containingNullsName =  selectedColumnDataSet.where(selectedColumnDataSet.col("name").isNull());
				containingNullsName.show();
				System.out.println("Dataset<Row> containingNullsName: "+containingNullsName.count());*/
				
				/*Dataset<Row> containingNullsRetweetCounnt=  selectedColumnDataSet.where(selectedColumnDataSet.col("retweet_count").isNull());
				containingNullsRetweetCounnt.show();
				System.out.println("Dataset<Row> containingNullsRetweetCounnt: "+containingNullsRetweetCounnt.count());
				
				Dataset<Row> containingNullsSidebarColor=  selectedColumnDataSet.where(selectedColumnDataSet.col("sidebar_color").isNull());
				containingNullsSidebarColor.show();
				System.out.println("Dataset<Row> containingNullsSidebarColor: "+containingNullsSidebarColor.count());
				
				
				Dataset<Row> containingNullsText=  selectedColumnDataSet.where(selectedColumnDataSet.col("text").isNull());
				containingNullsText.show();
				System.out.println("Dataset<Row> containingNullsText: "+containingNullsText.count());
				
				
				Dataset<Row> containingNullstweet_count=  selectedColumnDataSet.where(selectedColumnDataSet.col("tweet_count").isNull());
				containingNullstweet_count.show();
				System.out.println("Dataset<Row> containingNullstweet_count: "+containingNullstweet_count.count());*/
				
				
				
				// drop the rows containing any null or NaN values
				selectedColumnDataSet = selectedColumnDataSet.na().drop();
				System.out.println("Droping Rows containing null in csv: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());
				selectedColumnDataSet.show();
				
				//Label - HextoLong - process to convert it to a double variable
				sparkSession.udf().register("toHexLong", hexToLong, DataTypes.StringType);
				selectedColumnDataSet = selectedColumnDataSet.withColumn("link_color_long", callUDF("toHexLong", selectedColumnDataSet.col("link_color"))).drop("link_color");
				selectedColumnDataSet = selectedColumnDataSet.withColumn("sidebar_color_long", callUDF("toHexLong", selectedColumnDataSet.col("sidebar_color"))).drop("sidebar_color");
				System.out.println("After HexToLong csv: "+selectedColumnDataSet.columns().length+" : "+selectedColumnDataSet.count());
				selectedColumnDataSet.printSchema();
				selectedColumnDataSet.show();
				
				 
				
				
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
				traindata = traindata.select(traindata.col("_unit_id"),traindata.col("gender"), traindata.col(selectedColumn), traindata.col(selectedColumn2));
				traindata.show();
				
				testdata = testdata.select(traindata.col("_unit_id"),testdata.col("gender"), testdata.col(selectedColumn),traindata.col(selectedColumn2));
				testdata.show();

				// transform the String data into categorical variables
				StringIndexer indexer1 = new StringIndexer()
						.setInputCol("gender")
						.setOutputCol("gender1");
				StringIndexer indexer3 = new StringIndexer()
						.setInputCol("profile_yn")
						.setOutputCol("profile_yn1");
				
				// Configure an ML pipeline, which consists of multiple stages: indexer, tokenizer, hashingTF, idf, lr/rf etc 
				// and labelindexer.		
				//Relabel the target variable
				StringIndexerModel labelindexer = new StringIndexer()
						.setInputCol("gender")
						.setOutputCol("label").setHandleInvalid("keep").fit(traindata);
				
				// Tokenize the input text
				Tokenizer tokenizer = new Tokenizer()
						.setInputCol(selectedColumn)
						.setOutputCol("tokenizerwords");
				//Dataset<Row> df1Tokenizer = tokenizer.transform(traindata);
				
				// Remove the stop words
				StopWordsRemover remover = new StopWordsRemover()
						.setInputCol(tokenizer.getOutputCol())
						.setOutputCol("filtered");
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
				
				
				
				Pipeline pipeline = new Pipeline().setStages(new PipelineStage [] {tokenizer, remover,hashingTF, idf});
				PipelineModel model = pipeline.fit(traindata);
				Dataset<Row> df1IdfSet = model.transform(traindata);
				df1IdfSet.show();
				System.out.println("df1IdfSet Row Count  ::"+df1IdfSet.count());
				
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
				
				Pipeline pipeline2 = new Pipeline().setStages(new PipelineStage [] {tokenizer, remover,hashingTF, idf});
				PipelineModel model2 = pipeline2.fit(traindata);
				Dataset<Row> df1IdfSet2 = model2.transform(traindata);
				
				//df1IdfSet2.show();
				
				System.out.println("After second tokenizwer,and hashingTF:");
				
				df1IdfSet2 = df1IdfSet2.drop("gender").drop("description").drop("text");
				df1IdfSet2.show();
				System.out.println("df1IdfSet2 Row Count  ::"+df1IdfSet2.count());
				System.out.println("Perform Inner Join Operation");
				Dataset<Row> df1IdfSetJoined = df1IdfSet.join(df1IdfSet2, "_unit_id");
				df1IdfSetJoined.printSchema();
				df1IdfSetJoined.show();
				System.out.println("df1IdfSetJoined Row Count  ::"+df1IdfSetJoined.count());
				
				// Now Assemble Vectors
				VectorAssembler assembler = new VectorAssembler()
				        .setInputCols(new String[]{"features1", "features2"})
				        .setOutputCol("features");

				//Dataset<Row> assembledFeatures = assembler.transform(df1IdfSetJoined);
				
				//assembledFeatures.show();
				
				/*String idfColumnn1 = idf.getOutputCol();
				String idfColumnn2 = idf2.getOutputCol();*/

				
				
				/*VectorAssembler assembler = new
						VectorAssembler().setInputCols(new String[]{idfColumnn1,idfColumnn2}).setOutputCol("features");*/
				
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
						.setStages(new PipelineStage[] {labelindexer, tokenizer, remover,hashingTF,idf,tokenizer2,remover2,hashingTF2,idf2,assembler,rf,labelConverter});	
				// Fit the pipeline to training documents.
				PipelineModel modelRF = pipelineRF.fit(traindata);		
				// Make predictions on test documents.
				Dataset<Row> predictionsRF = modelRF.transform(testdata);
				System.out.println("Predictions from Random Forest Model are:");
				predictionsRF.show(10);

				// Create and Run Decision Tree Pipeline
				Pipeline pipelineDT = new Pipeline()
						.setStages(new PipelineStage[] {labelindexer, tokenizer, remover,hashingTF, idf,tokenizer2,remover2, hashingTF2,idf2,assembler,dt,labelConverter});	
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
