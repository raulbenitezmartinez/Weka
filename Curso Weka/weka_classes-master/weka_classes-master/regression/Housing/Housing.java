import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.Instance;

public class Housing{
    private String inputFilename;
    private Instances dataset;
    private Instance testInstance;

    public Housing(String inputFilename){
	this.inputFilename = inputFilename;
    }
    
    private void readData() throws Exception{
	// Leemos algun archivo arff, csv, ...
	dataset = DataSource.read(inputFilename);
	// Establecemos el ultimo attributo como el attributo a predecir
	dataset.setClassIndex(dataset.numAttributes() - 1);

	testInstance = dataset.instance(dataset.size() - 1);
	dataset.delete(dataset.size()-1);
	// Podemos ver el data set con System.out.println(dataset)
    }

    public void train() throws Exception{
	readData();

	// Entrenamos con ZeroR
	Classifier baseline = new ZeroR();
	baseline.buildClassifier(dataset);
	System.out.println(baseline);

	// Entrenamos con SimpleLinearRegression
	Classifier slr = new SimpleLinearRegression();
	slr.buildClassifier(dataset);
	System.out.println(slr);

	double price_slr = slr.classifyInstance(testInstance);
	System.out.println("Para " + testInstance + " se predice: " + price_slr);
	
	// Entrenamos con LinearRegression (default params)
	Classifier linear = new LinearRegression();
	linear.buildClassifier(dataset);
	System.out.println(linear);

	double price_linear = linear.classifyInstance(testInstance);
	System.out.println("Para " + testInstance + " se predice: " + price_linear);
    }
    
    public static void main(String[] args){
	String inputFilename = args[0];
	
	Housing housing = new Housing(inputFilename);
	try{
	    housing.train();
	}catch(Exception e){
	    System.err.println("Algo sucedió!");
	    System.err.println(e.getMessage());
	}
    }
}
