
package upgrad;

import static org.apache.spark.sql.functions.callUDF;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.stat.Correlation;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

public class ReadProcessDataModel {
	private static UDF1 percentageSplitTest = new UDF1<String, String>() {
		public String call(final String str) throws Exception {
			return str.split("%")[0];
		}
	};
	
	public void executeSteps() {
		//log messages set to output only error messages
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		System.setProperty("hadoop.home.dir", "C:\\spark\\bin");
		SparkSession sparkSession = SparkSession
				.builder().master("local[2]")
				.getOrCreate();
		
		String path = "data/gender-classifier.csv";
		
		// Read the file as a training dataset
		Dataset<Row> csv = sparkSession.read().
				format("csv").
				option("header","true").
				option("ignoreLeadingWhiteSpace",false). // you need this
				option("ignoreTrailingWhiteSpace",false).load(path);
		

		System.out.println("Dataset<Row> csv: "+csv.columns().length+" : "+csv.count());
		csv.show(10);
		
		// drop the columns which are mostly empty
		String[] dropCol = {"id","member_id","url","desc","zip_code","addr_state","earliest_cr_line","issue_d","last_pymnt_d",
				"next_pymnt_d","last_credit_pull_d","application_type",
				"annual_inc_joint","dti_joint","verification_status_joint","open_acc_6m",
				"open_act_il","open_il_12m","open_il_24m","mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m",
				"open_rv_24m","max_bal_bc","all_util","inq_fi","total_cu_tl","inq_last_12m",
				"revol_bal_joint","sec_app_earliest_cr_line","sec_app_inq_last_6mths","sec_app_mort_acc",
				"sec_app_open_acc","sec_app_revol_util","sec_app_open_act_il","sec_app_num_rev_accts",
				"sec_app_chargeoff_within_12_mths","sec_app_collections_12_mths_ex_med",
				"sec_app_mths_since_last_major_derog","hardship_flag","hardship_type","hardship_reason","hardship_status",
				"deferral_term","hardship_amount","hardship_start_date",
				"hardship_end_date","payment_plan_start_date","hardship_length","hardship_dpd","hardship_loan_status",
				"orig_projected_additional_accrued_interest","hardship_payoff_balance_amount","hardship_last_payment_amount",
				"disbursement_method","debt_settlement_flag","debt_settlement_flag_date",
				"settlement_status","settlement_date","settlement_amount","settlement_percentage","settlement_term","pymnt_plan",
				"mths_since_last_record","mths_since_last_major_derog","mths_since_recent_bc_dlq",
				"mths_since_recent_inq","mths_since_recent_revol_delinq","mths_since_last_delinq"};
		for(String column: dropCol) {
			csv = csv.drop(column);
		}
		System.out.println("Droping columns csv: "+csv.columns().length+" : "+csv.count());
		csv.show(10);

		// drop the rows containing any null or NaN values
		csv = csv.na().drop();
		System.out.println("Droping Rows containing null csv: "+csv.columns().length+" : "+csv.count());
		csv.show(10);
		
		//TO SELECT ONLY A SAMPLE OF DATA TO WORK WITH
		csv = csv.sample(true, 0.03).limit(10000);
		System.out.println(csv.count());
		
		// transform the String data into categorical variables
		StringIndexer indexer1 = new StringIndexer()
				.setInputCol("grade")
				.setOutputCol("grade1");
		StringIndexer indexer2 = new StringIndexer()
				.setInputCol("sub_grade")
				.setOutputCol("sub_grade1");
		StringIndexer indexer3 = new StringIndexer()
				.setInputCol("emp_title")
				.setOutputCol("emp_title1");
		StringIndexer indexer4 = new StringIndexer()
				.setInputCol("emp_length")
				.setOutputCol("emp_length1");
		StringIndexer indexer5 = new StringIndexer()
				.setInputCol("home_ownership")
				.setOutputCol("home_ownership1");
		StringIndexer indexer6 = new StringIndexer()
				.setInputCol("verification_status")
				.setOutputCol("verification_status1");
		StringIndexer indexer7 = new StringIndexer()
				.setInputCol("loan_status")
				.setOutputCol("loan_status1");
		StringIndexer indexer8 = new StringIndexer()
				.setInputCol("purpose")
				.setOutputCol("purpose1");
		StringIndexer indexer9 = new StringIndexer()
				.setInputCol("title")
				.setOutputCol("title1");
		StringIndexer indexer10 = new StringIndexer()
				.setInputCol("initial_list_status")
				.setOutputCol("initial_list_status1");
		StringIndexer indexer11 = new StringIndexer()
				.setInputCol("term")
				.setOutputCol("term1");
		List<StringIndexer> stringIndexer = Arrays.asList(indexer1, indexer2, indexer3, 
				indexer4, indexer5, indexer6, indexer7, indexer8, indexer9, indexer10, indexer11);
		for(StringIndexer indexer:stringIndexer) {
			csv = indexer.fit(csv).transform(csv);
		}
		System.out.println("Before Transforming csv: "+csv.columns().length+" : "+csv.count());
		csv.show(10);
		
		String[] dropCol1 = {"grade","initial_list_status","title","purpose","loan_status","verification_status",
				"home_ownership","emp_length","emp_title","sub_grade","term"};
		for(String col: dropCol1) {
			csv = csv.drop(col);
		}
		System.out.println("After Transforming csv: "+csv.columns().length+" : "+csv.count());
		csv.show(10);

		
		//Label - int_rate_final - process to convert it to a double variable
		sparkSession.udf().register("toPerSplit", percentageSplitTest, DataTypes.StringType);
		csv = csv.withColumn("int_rate_split", callUDF("toPerSplit", csv.col("int_rate"))).drop("int_rate");
		csv = csv.withColumn("revol_util_split", callUDF("toPerSplit", csv.col("revol_util"))).drop("revol_util");
		System.out.println("After splitting csv: "+csv.columns().length+" : "+csv.count());
		csv.show(10);

		// Converting String to double values
		String[] columns = csv.columns();
		for(String column:columns) {
			csv = csv.withColumn(column+"_final", csv.col(column).cast(DataTypes.DoubleType)).drop(column);
		}
		System.out.println("After Converting to double csv: "+csv.columns().length+" : "+csv.count());
		csv.show(10);
		
		// Modeling 
		// Create the feature vector using VectorAssembler
		Dataset<Row> featureCsv = csv.drop("int_rate_split_final");
		System.out.println("featureCsv: ");
		featureCsv.show(10);

		String[] inputCols = featureCsv.columns();
		VectorAssembler assembler = new
				VectorAssembler().setInputCols(inputCols).setOutputCol("features");

		Dataset<Row> featureVector = assembler.transform(csv);
		System.out.println("featureVector: ");
		featureVector.show(10);

		// Create the final dataset with "features" and "label"
		Dataset<Row> target = featureVector.select(featureVector.col("features"), 
				featureVector.col("int_rate_split_final")).withColumnRenamed("int_rate_split_final", "label");
		System.out.println("target: ");
		target.show(10);

		// Summary Statistics and Correlation Analysis 
		// Correlation Analysis
		Row r1 = Correlation.corr(target, "features").head();
		System.out.println("Pearson correlation matrix:\n" + r1.get(0).toString());
		
		// Split data into Training and testing
		Dataset<Row>[] dataSplit = target.randomSplit(new double[] { 0.7, 0.3 });
		Dataset<Row> trainingData = dataSplit[0];
		Dataset<Row> testData = dataSplit[1];

		System.out.println("trainingData: ");
		trainingData.show(10);
		System.out.println("testData: ");
		testData.show(10);

	
		
		
		
		// Run the Grid Search to find the best parameters
		double[] regParam = new double[] {0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15};
		double[] elasticParam = new double[] {0.3,0.4,0.5,0.6,0.7,0.8};

// Commented out this part as it takes lot of computation power and time. Might be difficult to run.
		// for(double reg:regParam) {
		// 	for(double elastic:elasticParam) {
		// 		//Define the model
		// 		LinearRegression lr = new LinearRegression()
		// 				.setMaxIter(10)
		// 				.setRegParam(reg)
		// 				.setElasticNetParam(elastic);
		// 		// Fit the model
		// 		LinearRegressionModel lrModel = lr.fit(trainingData);
		// 		// See the summary
		// 		LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
		// 		//Print the model characteristics
		// 		System.out.println("Linear Model => regParam: "+reg+" elasticParam: "+elastic);
		// 		System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
		// 		System.out.println("r2: " + trainingSummary.r2());			
		// 	}
		// }

		System.out.println("==============================================>");
		
		// Testing models using Best Parameters as per R2 and RMSE
		LinearRegression lr = new LinearRegression()
				.setMaxIter(10)
				.setRegParam(0.02)
				.setElasticNetParam(0.5);

		LinearRegressionModel lrModel = lr.fit(trainingData);

		Dataset<Row> results = lrModel.transform(testData);

		// Print the coefficients and intercept for logistic regression
		System.out.println("Coefficients: "
				+ lrModel.coefficients() + " Intercept: " + lrModel.intercept());

		RegressionEvaluator evaluator = new RegressionEvaluator()
				.setMetricName("rmse")
				.setLabelCol("label")
				.setPredictionCol("prediction");
		Double rmse = evaluator.evaluate(results);
		System.out.println("Root-mean-square error for the Best Fit Model using RegParam(0.02) and ElasticNetParam(0.5) => " + rmse);

		System.out.println("==============================================>");

		// Testing the normal Regression with out any regularisation parameters
		LinearRegression lr2 = new LinearRegression()
				.setMaxIter(10);

		LinearRegressionModel lrModel2 = lr2.fit(trainingData);

		LinearRegressionTrainingSummary trainingSummary2 = lrModel2.summary();

		System.out.println("Linear Model => default params ===> ");

		// Print the coefficients and intercept for linear regression
		System.out.println("Coefficients: "
				+ lrModel2.coefficients() + " Intercept: " + lrModel2.intercept());

		System.out.println("RMSE: " + trainingSummary2.rootMeanSquaredError());
		System.out.println("r2: " + trainingSummary2.r2());

		Dataset<Row> results2 = lrModel2.transform(testData);

		RegressionEvaluator evaluator2 = new RegressionEvaluator()
				.setMetricName("rmse")
				.setLabelCol("label")
				.setPredictionCol("prediction");
		Double rmse2 = evaluator2.evaluate(results2);
		System.out.println("Root-mean-square error for the Model with default values" + rmse2);
		
		
		
		
	}
}
