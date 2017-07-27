import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

public class Iris{
    private String filename;
    private Instances dataset;
    
    public Iris(String filename){
	this.filename = filename;
    }
    private void readData() throws Exception{
	dataset = DataSource.read(filename);
	dataset.setClassIndex(dataset.numAttributes()-1);
    }
    public void train() throws Exception{
	readData();

	// Entrenamos con Logistic Regression
	Classifier logit = new Logistic();
	logit.buildClassifier(dataset);
	System.out.println(logit);
    }
    
    public static void main(String[] args){
	try{
	    Iris iris = new Iris(args[0]);
	    iris.train();
	}catch(Exception e){
	    System.err.println(e.getMessage());
	}
    }
}
