import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;
import java.util.Random;

public class Cvps{
    private String filename;
    private long seed;
    private Instances dataset;

    public Cvps(String filename, long seed){
	this.filename = filename;
	this.seed = seed;
    }
    private void readData() throws Exception{
	dataset = DataSource.read(filename);
	dataset.setClassIndex(dataset.numAttributes() - 1);
    }
    private void train() throws Exception{
	readData();

	// Build classifier with k=5
	IBk knn = new IBk();
	knn.setOptions(Utils.splitOptions("-K 5"));

	Evaluation eval = new Evaluation(dataset);
	eval.crossValidateModel(knn, dataset, 10, new Random(seed));
	System.out.println("Without parameter optimization");
	System.out.println(eval.toSummaryString());

	// Search optimal parameters
	CVParameterSelection ps = new CVParameterSelection();
	ps.setClassifier(new IBk());
	ps.addCVParameter("K 1 10 10");
	ps.buildClassifier(dataset);
	System.out.println("Optimal parameter found");
	System.out.println(ps.toSummaryString());

	// Build classifier with optimal parameters
	knn = new IBk();
	knn.setOptions(ps.getBestClassifierOptions());

	eval = new Evaluation(dataset);
	eval.crossValidateModel(knn, dataset, 10, new Random(seed));
	System.out.println("With optimal params");
	System.out.println(eval.toSummaryString());
	
    }
    public static void main(String[] args){
	long seed = 152;
	Cvps cvps = new Cvps(args[0], seed);

	try{
	    cvps.train();
	}catch(Exception e){
	    System.err.println("Error: " + e.getMessage());
	    e.printStackTrace();
	}
    }
}
