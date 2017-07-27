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

//Para ejecutar: 	javac BIYML.java
//					java -cp weka.jar;. BIYML
//weka.jar se refiere a Weka 3.8


import weka.classifiers.Evaluation;//Clase para evaluar los clasificadores.
import weka.classifiers.evaluation.NominalPrediction;

public class BIMLCross {
	
	public static void main(String[] args) throws Exception {

		//#########################################
		//STEP 1: Express the problem with features
		//#########################################
		
		//String archivo = "C:\\Users\\raul\\Google Drive\\TESIS-BI-ML\\Weka\\WekaPracticas\\3_Codificacion_Tesis\\datos.arff";
		String archivo = "C:\\Users\\tecnologia\\Weka\\WekaPracticas\\3_Codificacion_Tesis\\datos.arff";		
		
		BufferedReader datos = null;
		Instances ejemplos = null;
 
		try{
			
			//Se lee el archivo ARFF.
			datos = new BufferedReader(new FileReader(archivo));
			
			//Se instancian los ejemplos.
			ejemplos = new Instances(datos);
			
			//Se configura el atributo class.
			ejemplos.setClassIndex(ejemplos.numAttributes() - 1);
			
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
			new ZeroR(),
			//TREES.			
			new DecisionStump(),
			//new HoeffdingTree(),
			new J48(),
			new LMT(),
			new RandomForest(),
			new RandomTree(),
			new REPTree(),
		};
		
		
		//Random rand = new Random(seed);   // create seeded number generator
		//randData = new Instances(data);   // create copy of original data
		//randData.randomize(rand);         // randomize data with number generator
		
		//In case your data has a nominal class and you wanna perform stratified cross-validation:
		//randData.stratify(folds);
		
		//Now, normally you would want to do a cross-validation and do this:
		//for (int n = 0; n < folds; n++) {
		//	Instances train = randData.trainCV(folds, n);
		//	Instances test = randData.testCV(folds, n);
		//}
		
		
		//Nuestro porcentaje aceptado sera el que genera el algoritmo OneR.
		Classifier modelo = new OneR();
		modelo.buildClassifier(ejemplos);
		
		//Se testea el modelo. Se testea con Cross Validation, para k=10.
		long seed = 1; //Default: 1
		Evaluation elTest = new Evaluation(ejemplos);
		elTest.crossValidateModel(modelo, ejemplos, 10, new Random(seed));
		porcentaje_aceptado = elTest.pctCorrect();

		
		//Se ejecuta para cada modelo. Utilizaremos 4 clasificadores: J48, PART, DecisionTable, DecisionStump.
		for (int j=0; j<modelos.length; j++){
			
			//IMPORTANTE: Todo lo anterior fue solo para poder construir uno o varios clasificadores.
			//Construimos un clasificador Naive Bayes.
			modelo = modelos[j];
			modelo.buildClassifier(ejemplos);
			
			//find the name of the first attribute
			//String name=ejemplos.attribute(0).name(); 
			//System.out.println(name); //look at new name
			
			//IMPORTANTE: Aqui se obtendra el porcentaje de aciertos del clasificador, para nuestros datos de testeo.
			
			//Se testea el modelo. Se testea sobre el mismo conjunto de entrenamiento.
			//Evaluation elTest = new Evaluation(ejemplos);
			//elTest.evaluateModel(modelo, ejemplos);
			
			//###########################
			//STEP 3: Test the classifier
			//###########################
			//Now that we create and trained a classifier, let’s test it. To do so, we need an evaluation module 
			//(weka.classifiers.Evaluation) to which we feed a testing set (see section 2, since the testing set 
			//is built like the training set).
			
			//Se testea el modelo. Se testea con Cross Validation, para k=10.
			elTest = new Evaluation(ejemplos);
			elTest.crossValidateModel(modelo, ejemplos, 10, new Random(seed));
			
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