package lr;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.MaxAbsScalerModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class SparkLinearRegression {
	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		SparkSession sparkSession = SparkSession.builder()
				.appName("SparkSQL")
				.master("local[*]")
				.getOrCreate();



		Dataset<Row> df1 = sparkSession.read().option("header", true).option("inferschema", true).csv("data/auto-miles-per-gallon-Raw.csv");
		//Reading Data from a CSV file //Inferring Schema and Setting Header as True

	//	df1.show(); //Displaying samples


		//****************************************** Handling missing values ******************************************************************

		//Casting MPG and HORSEPOWER from String to Double
		Dataset<Row> df2 = df1.selectExpr("cast(MPG as double ) MPG", "CYLINDERS","DISPLACEMENT",
				"cast(HORSEPOWER as double) HORSEPOWER","WEIGHT", 
				"ACCELERATION","MODELYEAR","NAME");


		//******************************************Replace missing values with approximate mean values*************************************

		//Imputer method automatically replaces null values with mean values.
		Imputer imputer = new Imputer()
				.setInputCols(new String[]{"MPG","HORSEPOWER"})
				.setOutputCols(new String[]{"MPG-Out","HORSEPOWER-Out"});

		ImputerModel imputeModel = imputer.fit(df2); //Fitting DataFrame into a model
		Dataset<Row> df4=imputeModel.transform(df2); //Transforming the DataFrame
		
		//Removing unnecessary columns
		Dataset<Row> df5 =df4.drop(new String[] {"MPG","HORSEPOWER"});

		//*******************************************Statistical Data Analysis*************************************************************

		StructType autoSchema = df5.schema(); //Inferring Schema



		for ( StructField field : autoSchema.fields() ) {	//Running through each column and performing Correlation Analysis
			if ( ! field.dataType().equals(DataTypes.StringType)) {
				System.out.println( "Correlation between MPG-Out and " + field.name()
				+ " = " + df5.stat().corr("MPG-Out", field.name()) );
			}
		}


		//****************************************Assembling the Vector and Label************************************************************

		//Renaming MPG-Out as lablel
		Dataset<Row> df6= df5.select(col("MPG-Out").as("label"),col("CYLINDERS"),col("WEIGHT"),col("HORSEPOWER-Out"),col("DISPLACEMENT"));

		//Assembling the features in the dataFrame as Dense Vector
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"CYLINDERS","WEIGHT","HORSEPOWER-Out","DISPLACEMENT"})
				.setOutputCol("features");

		Dataset<Row> LRdf = assembler.transform(df6).select("label","features");	

		//*********************************************Scaling the Vector***********************************************************************

		//Scaling the features between 0-1
		MaxAbsScaler scaler = new MaxAbsScaler() //Performing MaxAbsScaler() Transformation
				.setInputCol("features")
				.setOutputCol("scaledFeatures");

		// Building and Fitting in a MaxAbsScaler Model
		MaxAbsScalerModel scalerModel = scaler.fit(LRdf);

		// Re-scale each feature to range [0, 1].
		Dataset<Row> scaledData = scalerModel.transform(LRdf);

		//*********************************************Normalizing the Vector*********************************************************************

		//Normalizing the vector. Converts vector to a unit vector
		Normalizer normalizer = new Normalizer() //Performing Normalizer() Transformation
				.setInputCol("scaledFeatures")
				.setOutputCol("normFeatures")
				.setP(2.0);

		Dataset<Row> NormData = normalizer.transform(scaledData);
	
		//**********************************************Prepare for Linear Regression************************************************

		//Splitting the DataSet into training and testing
		// We split the data into training data and testing data. 
		// We use trainingData to train the model and use testingData to evaluate the model.
		
		//We use random split method to split the 80% (indicated by 0.8 in the brackets) of the data for trianing purpose 
		// and 20% for testing purpose. The random split method randomly splits the data, so when you execute the code multiple times
		//  the end results may differ. 
		// This is because the data for training the model is not constant and varies everytime we use randomSplit method
		Dataset<Row>[] dataSplit = NormData.randomSplit(new double[]{0.8, 0.2});
		Dataset<Row> trainingData = dataSplit[0];	//Training Data  0.8 --> 80%
		Dataset<Row> testingData = dataSplit[1];	//Testing Data        0.2 --> 20%

		//Setting up the Model properties
		// 1) Here we indicate what is input feature set column name in the dataset to the model.
		// so that model can use it to train itself.
		// 2) Also we indicate what is the label or dependent variable.So that model knows what is the output for a given input
		// 3) we use setMaxIter() method to stop the machine learning algorithm after particular number of iterations. 
		// You will learn how it is used later in this module. For now you can proceed with using this method also. 
		LinearRegression lr = new LinearRegression()	
				.setLabelCol("label")								
				.setFeaturesCol("normFeatures").setMaxIter(200);	//Performing the Iterations to improve model efficiency
		
	//**********************************************Performing for Linear Regression************************************************		
		
		//Now when we call fit method the model trains itself using the input column data, output column data
		//and properties defined above.
		LinearRegressionModel lrm = lr.fit(trainingData);	//Estimate or Build the model based on the Training Data

		//Print out coefficients of input variables and intercept, which are persent in the general equation, for our 
		// newly built model LR (Linear Regression Model)

		System.out.println("Coefficients: "+lrm.coefficients() + " Intercept: " + lrm.intercept());	//Printing the Coefficient and Intercept

		//Testing data by using linearRegressionModel
		//Predicting the values for our Testing Data based on our Linear Regression Model
		//Now here when we call transform method on testing data, 
		// the model will try to predict the output variable base on the input present in testing data
		// Also note that whenever we use transform method of spark library, the resulting dataset consists of the old cols and appended new columns

		Dataset<Row> predictionValues = lrm.transform(testingData);								

		//    Viewing schema and results of the dataset results
		predictionValues.printSchema();
		predictionValues.show();

		//**********************************************Evaluating the Model*********************************************************		
	// Here we use RegressionEvaluator to evaluate the model. You can use R-square metric or RMSE (root mean square error metric)
	// here we used r-square metric. If R-square (r2) is nearly 1 indicates that our model has performed very well, 
	// if r-square (r2) is approaching zero this indicates that model is performing poorly
	//  To know more on R-square or RMSE go through the optional content in session 3 or additional links provided
	//For the evaluator we indicate the actual value by setLabelCol() method and we indicate prediction col by setPredictionCol()
	// Finally we direct the model to use r2 metric with help of method setMetricName.
		RegressionEvaluator evaluator = new RegressionEvaluator()	//Evaluating the model 
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("r2");
		
	//	R2 is a statistic measure that will give some information about the goodness of fit of a model. 
	//	In regression, the R2 serves as a statistical measure of how well the regression predictions approximate the real data points. 
	//	An R2 indicates that the regression predictions perfectly fit the data.
		
      double r2 = evaluator.evaluate(predictionValues);		//Calculating the coefficient of determination which is r-square or r2 (all mean the same metric)
      System.out.println("R2 on test data = " + r2);

	}
}
