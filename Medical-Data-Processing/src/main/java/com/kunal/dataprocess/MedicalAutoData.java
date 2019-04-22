package com.kunal.dataprocess;

import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class MedicalAutoData {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		SparkSession sparkSession = SparkSession.builder()
				.appName("SparkML")
				.master("local[*]")
				.getOrCreate();
		
		Dataset<Row> df1 = sparkSession.read().option("header", true).option("inferschema", true).csv("data/Medical+Charges.csv");
		
		df1.show(); //Displaying samples
		df1.printSchema(); //Printing Schema
		df1.describe().show(); //Statistically summarizing about the data
		
		//****** Handling missing values ***********//
		
		//Casting age,charges and bmi from String to Double
		Dataset<Row> df2 = df1.select(col("age").cast("Double"), col("sex"),col("bmi").cast("Double"),
						col("children"),col("smoker"), 
						col("region"),col("charges").cast("Double"));
		
		System.out.println("*************************Casting columns********************************");
		df2.show(); //Displaying samples 
		df2.printSchema(); //Printing new Schema
		df2.describe().show();
		
		//****** Replacing missing values ***********//
		
		System.out.println("*******************Replacing records with missing values********************");
		
		//Imputer method automatically replaces null values with mean values.
		Imputer imputer = new Imputer()
				.setInputCols(new String[]{"age","bmi","charges"})
				.setOutputCols(new String[]{"age-Out","bmi-Out","charges-Out"});
		
		ImputerModel imputeModel = imputer.fit(df2); //Fitting DataFrame into a model
		Dataset<Row> df4=imputeModel.transform(df2); //Transforming the DataFrame
		df4.show();
		df4.printSchema();
		df4.describe().show(); //Describing the dataframe
		
		//******** Bukceting the data *******//
		
		//Removing unnecessary columns
		Dataset<Row> df5 =df4.drop(new String[] {"age","bmi","charges"});
		df5.printSchema();
		
		// Bucketize multiple columns at one pass.
	    double[][] splitsArray = {
	      {Double.NEGATIVE_INFINITY, 24, 35, 45, 55,Double.POSITIVE_INFINITY}
	      //{Double.NEGATIVE_INFINITY, -0.3, 0.0, 0.3, Double.POSITIVE_INFINITY}
	    };
	    
	    Bucketizer bucketizer1 = new Bucketizer()
	    		.setInputCols(new String[]{"age-Out"})
	    		.setOutputCols(new String[]{"age-Bucket"})
	    		.setSplitsArray(splitsArray);
	    
	 // Transform original data into its bucket index.
	    Dataset<Row> bucketedData1 = bucketizer1.transform(df5);
	    bucketedData1.show();
	    bucketedData1.printSchema();
	    bucketedData1.describe().show();

	}

}
