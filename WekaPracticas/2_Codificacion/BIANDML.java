import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;
import weka.core.Attribute;
import weka.core.Instances;//Clase para los ejemplos.
import weka.core.Instance;
import weka.core.FastVector;//Clase para agrupar las predicciones de la clase Evaluation.
import weka.classifiers.Classifier;//Clase para conjunto de clasificadores.
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.Evaluation;//Clase para evaluar los clasificadores.
import weka.classifiers.evaluation.NominalPrediction;

public class BIANDML {
	
	public static void main(String[] args) throws Exception {

		//#########################################
		//STEP 1: Express the problem with features
		//#########################################
		//This step corresponds to the engineering task needed to write an .arff file.
		//Let’s put all our features in a weka.core.FastVector.
		//Each feature is contained in a weka.core.Attribute object.
		//Here, we have two numeric features, one nominal feature (blue, gray, black) 
		//and a nominal class (positive, negative).
		
		int cantidad_de_atributos = 13;//12 atributos numericos y 1 atributo de clase.
		int cantidad_de_ejemplos = 100;//Cantidad de ejemplos para nuestro modelo.
		int cantidad_de_etiquetas = 2;//El atributo de clase tiene dos posibles valores: 0 o 1.
		int pos_atributo_clase = 12;//Weka indexara de 0 a 12 los 13 atributos. En el indice 12 estara el atributo de clase.
		
		//Declaramos 12 atributos numericos.
		Attribute cre_fid = new Attribute("cre_fid");
		Attribute cli_fie = new Attribute("cli_fie");
		Attribute tas_aba = new Attribute("tas_aba");
		Attribute tic_med = new Attribute("tic_med");
		Attribute cif_ven = new Attribute("cif_ven");
		Attribute mar_com = new Attribute("mar_com");
		Attribute rot_sto = new Attribute("rot_sto");
		Attribute coe_ren = new Attribute("coe_ren");
		Attribute cob_sto = new Attribute("cob_sto");
		Attribute dif_inv = new Attribute("dif_inv");
		Attribute umb_ren = new Attribute("umb_ren");
		Attribute roi = new Attribute("roi");
 
		//Declaramos el atributo de clase, para nuestro caso 1 (Comprar) o 0 (No comprar).
		FastVector val_atri_class = new FastVector(cantidad_de_etiquetas);
		val_atri_class.addElement("0");
		val_atri_class.addElement("1");
		Attribute dec_com = new Attribute("dec_com", val_atri_class);

		//Declaramos el vector de caracteristicas.
		FastVector atributos = new FastVector(cantidad_de_atributos);
		atributos.addElement(cre_fid);
		atributos.addElement(cli_fie);
		atributos.addElement(tas_aba);
		atributos.addElement(tic_med);
		atributos.addElement(cif_ven);
		atributos.addElement(mar_com);
		atributos.addElement(rot_sto);
		atributos.addElement(coe_ren);
		atributos.addElement(cob_sto);
		atributos.addElement(dif_inv);
		atributos.addElement(umb_ren);
		atributos.addElement(roi);		
		atributos.addElement(dec_com);
		
		//##########################
		//STEP 2: Train a Classifier
		//##########################
		//Training requires 1) having a training set of instances and 2) choosing a classifier.
		//Let’s first create an empty training set (weka.core.Instances).
		//We named the relation "Relacion decisiones de compras".
		//The attribute prototype is declared using the vector from step 1.
		//We give an initial set capacity of 100.
		//We also declare that the class attribute is the 13 one in the vector (see step 1)
		
		//Creamos un conjunto de entrenamiento vacio.
		Instances conjunto_entrenamiento = new Instances("Relacion decisiones de compras", atributos, cantidad_de_ejemplos);
		//Establecemos el indice del atributo clase.
		conjunto_entrenamiento.setClassIndex(pos_atributo_clase);
		
		//Llenamos el conjunto de entrenamiento con varias instancias (weka.core.Instance):
		//Se lee el archivo de datos.
		String archivo = "C:\\Users\\raul\\Google Drive\\TESIS-BI-ML\\Weka\\WekaPracticas\\2_Codificacion\\datos.txt";	
		BufferedReader datos = null;
		String linea = null;
		int contador_lineas = 0;
		int contador_elementos = 0;
		int sumador = 0;
		
		try{
			
			datos = new BufferedReader(new FileReader(archivo));
			
			//Creamos las instancias de ejemplos.
			while((linea = datos.readLine()) != null){
					
				Instance ejemplo = new Instance(cantidad_de_atributos);
				String[] splited = linea.split(",");
				
				for (int i = 0; i < cantidad_de_atributos - 1; i++){
					ejemplo.setValue(conjunto_entrenamiento.attribute(i),Double.parseDouble(splited[i]));
					contador_elementos++;
				}
				
				ejemplo.setValue(conjunto_entrenamiento.attribute(cantidad_de_atributos - 1),splited[cantidad_de_atributos - 1]);
				contador_elementos++;
				
				//IMPORTANTE: Aqui se agrega cada ejemplo de nuestro conjunto de entrenamiento.
				//Agregamos el ejemplo.
				conjunto_entrenamiento.add(ejemplo);
				contador_lineas++;
			}
			
		}catch(IOException e){
			System.err.println("Este es el problema: " + e);
			e.printStackTrace();
		}finally{
			try {
				datos.close();
			}catch (Exception e){
			}
		}
		
		if(contador_lineas == cantidad_de_ejemplos && contador_elementos == cantidad_de_atributos * cantidad_de_ejemplos){
			System.out.println("\nOK!, Cantidad correcta de datos.");
		}else{
			System.exit(0);
		}
		
		//Finalmente, elegimos un clasificador (classifier) (weka.classifiers.Classifier) y creamos el modelo. 
		//Por ejemplo: creamos un clasificador Naive Bayes (weka.classifiers.bayes.NaiveBayes).
		
		//IMPORTANTE: Todo lo anterior fue solo para poder construir uno o varios clasificadores.
		//Construimos un clasificador Naive Bayes.
		Classifier modeloNAIVEBAYES = (Classifier)new NaiveBayes();
		modeloNAIVEBAYES.buildClassifier(conjunto_entrenamiento);
		
		//###########################
		//STEP 3: Test the classifier
		//###########################
		//Now that we create and trained a classifier, let’s test it. To do so, we need an evaluation module 
		//(weka.classifiers.Evaluation) to which we feed a testing set (see section 2, since the testing set 
		//is built like the training set).
		
		//Se lee el archivo ARFF.
		String archivo_testing = "C:\\Users\\raul\\Google Drive\\TESIS-BI-ML\\Weka\\WekaPracticas\\2_Codificacion\\data.arff";
		BufferedReader datafile = null;
		
		try{
			
			//Cargamos el conjunto de testing.
			datafile = new BufferedReader(new FileReader(archivo_testing));
			Instances conjunto_testing = new Instances(datafile);
			
			//Establecemos el indice del atributo clase.
			conjunto_testing.setClassIndex(pos_atributo_clase);
			
			//IMPORTANTE: Aqui se obtendra el porcentaje de acierto del clasificador, para nuestros datos de testeo.
			//Se testea el modelo. Se testea sobre el mismo conjunto de entrenamiento.
			Evaluation elTest = new Evaluation(conjunto_entrenamiento);
			elTest.evaluateModel(modeloNAIVEBAYES, conjunto_testing);
			
			//Imprimimos el resultado del Weka explorer.
			System.out.println("\nRESUMEN: ");
			String resumen = elTest.toSummaryString();
			System.out.println(resumen);
 
			//Obtenemos la matriz de confusion.
			double[][] matriz_confusion = elTest.confusionMatrix();
			
			//Imprimimos la matriz de confusion.
			System.out.println("\nMATRIZ DE CONFUSION:\n");
			for (int i = 0; i < matriz_confusion.length; i++) {
				for (int j = 0; j < matriz_confusion[0].length; j++) {
					System.out.print(matriz_confusion[i][j] + " ");
				}
				System.out.print("\n");
			}
			
		}catch(IOException e){
			System.err.println("\nEste es el problema: " + e);
			e.printStackTrace();
		}finally{
			try {
				datafile.close();
			}catch (Exception e){
			}
		}
		
		//##########################
		//STEP 4: Use the classifier
		//##########################
		//For real world applications, the actual use of the classifier is the ultimate goal. 
		//Here’s the simplest way to achieve that. Let’s say we’ve built an instance (named iUse) 
		//as explained in step 2:
		
		try{
			
			Instance nuevo_ejemplo = new Instance(cantidad_de_atributos);
				
			for (int i = 0; i < cantidad_de_atributos - 1; i++){
				nuevo_ejemplo.setValue(conjunto_entrenamiento.attribute(i),Double.parseDouble("0.9"));
			}
						
			//IMPORTANTE: Aqui se agrega cada nuevo_ejemplo de nuestro conjunto de entrenamiento.
			//Agregamos el nuevo_ejemplo.
			conjunto_entrenamiento.add(nuevo_ejemplo);			
			
			//Proceso de etiquetado del nuevo ejemplo.
			Instances etiquetar = new Instances(conjunto_entrenamiento);
			double nueva_etiqueta = modeloNAIVEBAYES.classifyInstance(conjunto_entrenamiento.lastInstance());
			etiquetar.lastInstance().setClassValue(nueva_etiqueta);
			
			//Imprimimos el resultado del Weka explorer.
			System.out.println("\nEL NUEVO EJEMPLO TIENE LA ETIQUETA: " + etiquetar.lastInstance());

		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}