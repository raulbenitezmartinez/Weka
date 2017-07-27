import java.io.BufferedReader;
import java.io.FileNotFoundException;
//import java.io.IOException;
//import java.io.FileReader;
//import java.io.File;
import java.util.List;
//import java.util.ArrayList;
//import java.util.HashSet;
//import java.util.Set;
//import java.util.Random;
import weka.core.converters.CSVLoader;
import weka.classifiers.rules.OneR;

//import weka.core.Attribute;
//import weka.core.Instances;//Clase para los ejemplos, tambien llamados instancias.
//import weka.core.Instance;
//import weka.core.FastVector;//Clase para agrupar las predicciones de la clase Evaluation.
import weka.classifiers.Classifier;//Clase para conjunto de clasificadores.

//BAYES.
//import weka.classifiers.bayes.BayesNet;
//import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.bayes.NaiveBayesMultinomial;
//import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
//import weka.classifiers.bayes.NaiveBayesUpdateable;

public class PRUEBA {
	
	public static void main(String[] args) throws Exception {
		
		Classifier modelo = new OneR();
		
		System.out.println("\nHola mundo");
		
	}
}