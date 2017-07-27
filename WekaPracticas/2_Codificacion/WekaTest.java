import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.core.Instances;//Clase para los ejemplos.
import weka.classifiers.Classifier;//Clase para conjunto de clasificadores.
import weka.classifiers.Evaluation;//Clase para evaluar los clasificadores.
import weka.core.FastVector;//Clase para agrupar las predicciones de la clase Evaluation.
import weka.classifiers.trees.J48;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.evaluation.NominalPrediction;

public class WekaTest {
	
	public static void main(String[] args) throws Exception {
		
		//Se lee el archivo ARFF.
		String archivo = "C:\\Users\\raul\\Desktop\\WekaPracticas\\2_Codificacion\\data.arff";
		BufferedReader datafile = readDataFile(archivo);
 
		//Se configura el atributo class.
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
 
		//Se utiliza cross validation con 10 folds.
		Instances[][] split = crossValidationSplit(data, 10);
 
		//Se obtienen dos arrays a partir de split: training y testing.
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		//Conjunto de classifiers.
		Classifier[] models = { 
			new J48(),//a decision tree
			new PART(), 
			new DecisionTable(),//decision table majority classifier
			new DecisionStump()//one-level decision tree
		};
 
		//Se ejecuta para cada modelo. Utilizaremos 4 clasificadores: J48, PART, DecisionTable, DecisionStump.
		for (int j=0; j<models.length; j++){
 
			//Juntamos cada grupo de predicciones, para el actual modelo, en un FastVector.
			FastVector predictions = new FastVector();
 
			//Para cada par trainingSplits-testingSplits, se obtiene un conjunto de predicciones.
			for (int i = 0; i < trainingSplits.length; i++) {
				
				//Aqui se evalua el clasificador.
				Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
				
				//Aqui se obtienen las predicciones.
				predictions.appendElements(validation.predictions());
 
				//Resumen de cada par trainingSplits-testingSplits.
				//System.out.println(models[j].toString());
			}
 
			//Se calcula la precisión global del clasificador.
			double accuracy = calculateAccuracy(predictions);
 
			//Se imprime la precision del clasificador en cuestion.
			System.out.println("Precision del modelo " + models[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracy)
					+ "\n---------------------------------");
		}
	}
	
	//Metodo que instancia un fichero.
	public static BufferedReader readDataFile(String filename){
		
		BufferedReader inputReader = null;
 
		try{
			inputReader = new BufferedReader(new FileReader(filename));
		}catch(FileNotFoundException ex){
			System.err.println("Archivo no encontrado: " + filename);
		}
 
		return inputReader;
	}

	//Metodo para obtener dos sub-conjuntos (training y testing), a partir de un conjunto de ejemplos (data).
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds){
		
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i=0; i<numberOfFolds; i++){
			
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
			
			//System.out.println(split[0][i].toString());
			//System.out.println(split[1][i].toString());
		}
 
		return split;
	}
	
	//Metodo para evaluar clasificadores, a partir de un trainingSet y de un testingSet.
	public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception{
		
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
 
	//Metodo para calcular la precision de un conjunto de predicciones.
	public static double calculateAccuracy(FastVector predictions){
		
		double correct = 0;
 
		for (int i=0; i<predictions.size(); i++){
			
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			
			if(np.predicted() == np.actual()){
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
}