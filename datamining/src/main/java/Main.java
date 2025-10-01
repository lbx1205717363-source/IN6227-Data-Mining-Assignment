import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.util.Random;

/**
 * IN6227 Data Mining - Assignment 1
 */
public class Main {

    public static void main(String[] args) {
        try {
            // Define the dataset file path
            String filePath = "src/main/resources/adult.data";

            // Load and preprocess the data
            System.out.println("Loading and preprocessing data...");
            Instances data = loadAndPreprocessData(filePath);
            System.out.println("Data loaded and preprocessed successfully.");
            System.out.println("--------------------------------------------------\n");


            // Initialize and evaluate the first model - Decision Tree (J48)
            System.out.println("===== Evaluating Model 1: Decision Tree (J48) =====");
            J48 j48Tree = new J48();

            // Hyperparameter Tuning
            // -C 0.25 is the confidence factor used for pruning. A smaller value results in more pruning.
            // -M 2 is the minimum number of instances required for each leaf node.
            j48Tree.setOptions(new String[]{"-C", "0.25", "-M", "2"});

            evaluateModel(j48Tree, data, "Decision Tree (J48)");
            System.out.println("--------------------------------------------------\n");


            // Initialize and evaluate the second model - Logistic Regression
            System.out.println("===== Evaluating Model 2: Logistic Regression =====");
            Logistic logistic = new Logistic();

            // Hyperparameter Tuning
            // -R 1.0E-8 sets the Ridge parameter.
            logistic.setOptions(new String[]{"-R", "1.0E-8"});

            evaluateModel(logistic, data, "Logistic Regression");
            System.out.println("--------------------------------------------------\n");

        } catch (Exception e) {
            System.err.println("An error occurred during the process:");
            e.printStackTrace();
        }
    }


    public static Instances loadAndPreprocessData(String filePath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        loader.setNoHeaderRowPresent(true);

        Instances data = loader.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        ReplaceMissingValues missingValuesFilter = new ReplaceMissingValues();
        missingValuesFilter.setInputFormat(data);
        data = Filter.useFilter(data, missingValuesFilter);

        NominalToBinary nominalToBinaryFilter = new NominalToBinary();
        nominalToBinaryFilter.setInputFormat(data);
        data = Filter.useFilter(data, nominalToBinaryFilter);

        return data;
    }

    /**
     * Evaluate a classifier using 10-fold cross validation and print performance metrics。
     *
     * @param model The classifier to be evaluated
     * @param data  Datasets used for evaluation
     * @param modelName Name of the model
     */
    public static void evaluateModel(Classifier model, Instances data, String modelName) throws Exception {
        // Use Weka's Evaluation class for evaluation
        Evaluation eval = new Evaluation(data);

        // Record the start time of cross validation
        long startTime = System.currentTimeMillis();

        eval.crossValidateModel(model, data, 10, new Random(1));

        // Record the end time and calculate the elapsed time
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;

        System.out.println("--- Performance Metrics for " + modelName + " ---");
        // Get and print various indicators from the Evaluation object
        int positiveClassIndex = 1;

        System.out.println("Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));
        System.out.println("Precision (class '>50K'): " + String.format("%.4f", eval.precision(positiveClassIndex)));
        System.out.println("Recall (class '>50K'): " + String.format("%.4f", eval.recall(positiveClassIndex)));
        System.out.println("F1-Score (class '>50K'): " + String.format("%.4f", eval.fMeasure(positiveClassIndex)));
        System.out.println("Area Under ROC (AUC): " + String.format("%.4f", eval.areaUnderROC(positiveClassIndex)));

        System.out.println("\n--- Time Measurement ---");
        System.out.println("Total time for 10-fold Cross-Validation: " + executionTime + " ms");

        System.out.println("\n--- Weka Evaluation Summary ---");
        System.out.println(eval.toSummaryString());
    }
}