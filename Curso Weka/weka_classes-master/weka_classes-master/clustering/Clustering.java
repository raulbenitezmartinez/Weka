import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.EM;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;

public class Clustering{
    private String filename;
    private Instances dataset;
    private Instances datasetEval;

    public Clustering(String filename){
	this.filename = filename;
    }
    private void readData() throws Exception{
	// Definimos el dataset para evaluación
	// Mismos datos con el atributo a evaluar
	datasetEval = DataSource.read(filename);
	datasetEval.setClassIndex(datasetEval.numAttributes()-1);

	// Definimos el dataset para entrenamiento
	// Mismos datos sin el atributo a evaluar
	Remove filter = new Remove();
	filter.setAttributeIndices("" + datasetEval.numAttributes());
	filter.setInputFormat(datasetEval);
	dataset = Filter.useFilter(datasetEval, filter);
    }
    public void classesToClusterEvaluation() throws Exception {
	readData();

	ClusterEvaluation eval = new ClusterEvaluation();

	// Evaluamos K-means (seed=155)
	String kmOptions = "-N 3"; // Number of clusters
	SimpleKMeans km = new SimpleKMeans();
	km.setOptions(Utils.splitOptions(kmOptions));
	km.setSeed(155);
	km.buildClusterer(dataset);
	eval.setClusterer(km);
	eval.evaluateClusterer(datasetEval);
	
	System.out.println(eval.clusterResultsToString());

	// Entrenamos K-means (seed=551)
	km.setOptions(Utils.splitOptions(kmOptions));
	km.setSeed(551);
	km.buildClusterer(dataset);
	eval.setClusterer(km);
	eval.evaluateClusterer(datasetEval);

	System.out.println(eval.clusterResultsToString());

	// Entrenamos K-means++
	String kmppOptions = "-N 3 -init 1"; // 0 = random, 1 = k-means++, 2 = canopy, 3 = farthest first.
	SimpleKMeans kmpp = new SimpleKMeans();
	kmpp.setOptions(Utils.splitOptions(kmppOptions));
	kmpp.buildClusterer(dataset);
	eval.setClusterer(kmpp);
	eval.evaluateClusterer(datasetEval);

	System.out.println(eval.clusterResultsToString());

	// Entrenamos Expectation-Maximization
	String emOptions = "-N 3";
	EM em = new EM();
	em.setOptions(Utils.splitOptions(emOptions));
	em.buildClusterer(dataset);
	eval.setClusterer(em);
	eval.evaluateClusterer(datasetEval);

	System.out.println(eval.clusterResultsToString());

    }
    
    public static void main(String[] args){
	try{
	    Clustering c = new Clustering(args[0]);
	    c.classesToClusterEvaluation();
	}catch(Exception e){
	    System.err.println(e.getMessage());
	    e.printStackTrace();
	}
    }
}
