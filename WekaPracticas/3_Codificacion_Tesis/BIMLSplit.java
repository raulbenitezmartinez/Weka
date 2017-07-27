import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;
import java.io.File;
import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.Random;
import weka.core.converters.CSVLoader;
import weka.core.Attribute;
import weka.core.Instances;//Clase para los ejemplos, tambien llamados instancias.
import weka.core.Instance;
import weka.core.FastVector;//Clase para agrupar las predicciones de la clase Evaluation.

import weka.classifiers.Classifier;//Clase para conjunto de clasificadores.

//BAYES.
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesUpdateable;
//import weka.classifiers.bayes.ComplementNaiveBayes;
//import weka.classifiers.bayes.DMNBtext;
//import weka.classifiers.bayes.NaiveBayesSimple;

//FUNCTIONS.
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.SMO;
//import weka.classifiers.functions.LibLINEAR;
//import weka.classifiers.functions.LibSVM;
//import weka.classifiers.functions.RBFNetwork;

//RULES.
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
//import weka.classifiers.rules.ConjunctiveRule;
//import weka.classifiers.rules.DTNB;
//import weka.classifiers.rules.NNge;
//import weka.classifiers.rules.Ridor;

//TREES.
import weka.classifiers.trees.DecisionStump;
//import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.REPTree;
//import weka.classifiers.trees.BFTree;
//import weka.classifiers.trees.FT;
//import weka.classifiers.trees.J48graft;
//import weka.classifiers.trees.LADTree;
//import weka.classifiers.trees.NBTree;
//import weka.classifiers.trees.SimpleCart;
//import weka.classifiers.trees.UserClassifier;

//Para ejecutar: 	javac BIMLSplit.java
//					java -cp weka.jar;. BIMLSplit
//weka.jar se refiere a Weka 3.8


import weka.classifiers.Evaluation;//Clase para evaluar los clasificadores.
import weka.classifiers.evaluation.NominalPrediction;

public class BIMLSplit {
	
	public static void main(String[] args) throws Exception {

		//#########################################
		//STEP 1: Express the problem with features
		//#########################################
		
		//String archivo = "C:\\Users\\raul\\Google Drive\\TESIS-BI-ML\\Weka\\WekaPracticas\\3_Codificacion_Tesis\\datos.arff";
		String archivo = "C:\\Users\\tecnologia\\Weka\\WekaPracticas\\3_Codificacion_Tesis\\datos.arff";		
		
		BufferedReader datos = null;
		Instances ejemplos = null;
		Instances train = null;
		Instances test = null;
		long seed = 1; //Default: 1
		int trainSize;
		int testSize;
		
		try{
			
			//Se lee el archivo ARFF.
			datos = new BufferedReader(new FileReader(archivo));
			
			//Se instancian los ejemplos.
			ejemplos = new Instances(datos);
			
			//Se ramdomiza el dataset.
			ejemplos.randomize(new Random(seed));
			
			//Se realiza el split, se obtienen dos subconjunto: train y test.
			trainSize = (int) Math.round(ejemplos.numInstances() * 0.6);
			testSize = ejemplos.numInstances() - trainSize;
			train = new Instances(ejemplos, 0, trainSize);
			test = new Instances(ejemplos, trainSize, testSize);
			
			//Se configura el atributo class.
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);
			
		}catch(IOException e){
			System.err.println("Este es el problema: " + e);
			e.printStackTrace();
		}finally{
			try {
				datos.close();
			}catch (Exception e){
			}
		}
		
		//##########################
		//STEP 2: Train a Classifier
		//##########################
		
		double porcentaje_aceptado = 90.00;
		int coincidencias = 0;
		
		//Conjunto de clasificadores.
		Classifier[] modelos = {
			//BAYES.			
			new BayesNet(),
			new NaiveBayes(),
			//new NaiveBayesMultinomial(),
			//new NaiveBayesMultinomialUpdateable(),
			new NaiveBayesUpdateable(),
			//FUNCTIONS.			
			new Logistic(),
			new MultilayerPerceptron(),
			new SimpleLogistic(),
			new SMO(),
			//RULES.			
			new DecisionTable(),
			new JRip(),
			new PART(),
			new OneR(),
			//TREES.			
			new DecisionStump(),
			//new HoeffdingTree(),
			new J48(),
			new LMT(),
			new RandomForest(),
			new RandomTree(),
			new REPTree(),
		};
		
		//Nuestro porcentaje aceptado sera el que genera el algoritmo OneR.
		Classifier modelo = new ZeroR();
		modelo.buildClassifier(train);
		
		//Se testea el modelo. Se testea con Cross Validation, para k=10.
		Evaluation elTest = new Evaluation(test);
		elTest.evaluateModel(modelo, test);
		porcentaje_aceptado = elTest.pctCorrect();
		
		//Se ejecuta para cada modelo. Utilizaremos 4 clasificadores: J48, PART, DecisionTable, DecisionStump.
		for (int j=0; j<modelos.length; j++){
			
			//IMPORTANTE: Todo lo anterior fue solo para poder construir uno o varios clasificadores.
			//Construimos un clasificador Naive Bayes.
			modelo = modelos[j];
			modelo.buildClassifier(train);
			
			//find the name of the first attribute
			//String name=ejemplos.attribute(0).name(); 
			//System.out.println(name); //look at new name
			
			//###########################
			//STEP 3: Test the classifier
			//###########################
			//Now that we create and trained a classifier, let’s test it. To do so, we need an evaluation module 
			//(weka.classifiers.Evaluation) to which we feed a testing set (see section 2, since the testing set 
			//is built like the training set).
			
			//IMPORTANTE: Aqui se obtendra el porcentaje de aciertos del clasificador, para nuestros datos de testeo.
			//Se testea el modelo. Se testea con Cross Validation, para k=10.
			elTest = new Evaluation(test);
			elTest.evaluateModel(modelo, test);
			
			//Obtenemos la matriz de confusion.
			//double[][] matriz_confusion = elTest.confusionMatrix();
			
			//Imprimimos la matriz de confusion.
			//System.out.println("\nMATRIZ DE CONFUSION:\n");
			//for (int i = 0; i < matriz_confusion.length; i++) {
			//	for (int j = 0; j < matriz_confusion[0].length; j++) {
			//		System.out.print(matriz_confusion[i][j] + " ");
			//	}
			//	System.out.print("\n");
			//}
						
			if (elTest.pctCorrect() >= porcentaje_aceptado){
				
				coincidencias++;
				
				//Imprimimos el resultado del Weka explorer.
				String name = modelo.getClass().getName();
				System.out.println("\nRESUMEN: "+ name);
			
				//
				//System.out.println(elTest.correct());
				//System.out.println(elTest.pctCorrect());
			
				//
				String resumen = elTest.toSummaryString();
				String[] splited = resumen.split("\\r\\n|\\n|\\r");
				System.out.println(splited[1]);
				System.out.println(splited[2]);			
			}
		}
		
		System.out.println("\nPara un porcentaje aceptado de coincidencias del " + porcentaje_aceptado + "%, " + "se encontraron " + coincidencias + " algoritmos.");
		
	}
}