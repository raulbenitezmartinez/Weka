import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class Trees{
    private String filename;
    private Instances dataset;
    
    public Trees(String filename){
	this.filename = filename;
    }
    private void readData() throws Exception{
	dataset = DataSource.read(filename);
	dataset.setClassIndex(dataset.numAttributes()-1);
    }
    public void train() throws Exception{
	readData();

	// Entrenamos con ZeroR
	Classifier baseline = new ZeroR();
	baseline.buildClassifier(dataset);
	System.out.println(baseline);

	// Entrenamos con J48 (C45)
	Classifier j48 = new J48();
	j48.buildClassifier(dataset);
	System.out.println(j48);

	// Entrenamos con RandomTree
	int rtSeed = 123;
	RandomTree rt = new RandomTree();
	rt.setSeed(rtSeed);
	rt.buildClassifier(dataset);
	System.out.println(rt);
	
	// Guardamos el J48
	SerializationHelper.write("j48.model", j48);
	System.out.println("Model saved.");

	// Cargamos un modelo
	Classifier randomModel = (Classifier) SerializationHelper.read("j48.model");
	System.out.println();
	System.out.println("Model loaded:");
	System.out.println(randomModel);
    }
    
    public static void main(String[] args){
	Trees trees = new Trees(args[0]);
	try{
	    trees.train();
	}catch(Exception e){
	    System.err.println(e.getMessage());
	}
    }
}
