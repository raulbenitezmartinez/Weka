package Clases;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.StringWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Set;
import java.util.Random;
import java.text.DecimalFormat;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.core.Attribute;
import weka.core.Instances;//Clase para los ejemplos, tambien llamados instancias.
import weka.core.Instance;
import weka.core.FastVector;//Clase para agrupar las predicciones de la clase Evaluation.
import weka.classifiers.Classifier;//Clase para conjunto de clasificadores.
import weka.classifiers.Evaluation;//Clase para evaluar los clasificadores.
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

//BAYES.
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesUpdateable;

//FUNCTIONS.
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.SMO;

//RULES.
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;

//TREES.
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.REPTree;

//weka.jar se refiere a Weka 3.8

/** Un programa Java simple.
  * Que hace?.
  * @author Raul Benitez
  * @version 1
  */
public class BIML {
	
	private String filepath = "./Generados/";
    private String filename = "";
    private Instances instancias = null;
	private	Instances train = null;
	private	Instances test = null;
	//private	Instances test1 = null;
	//private	Instances test2 = null;
	private	int trainSize = 0;
	private	int testSize = 0;
	//private	int test1Size = 0;
	//private	int test2Size = 0;
	private long seed = 1; //Default = 1
	private int folds = 10;
	private int max_instancias = 151;
	private double porcentaje_split = 0.6;
	private List<String> lista_claves = null;
	private List<String> etiquetas = null;
	private String cabecera = "";
	private String extra_cabecera = "";
	private String extra_datos = "";
	private FileWriter fstream;
	private BufferedWriter out;
    
    public BIML(String fn, List<String> e, int max_inst){
		
		this.filepath = this.filepath + fn;
		this.filename = fn;
		this.etiquetas = e;
		this.max_instancias = max_inst;
		
		//Se instancian los ejemplos.
		instanciasDesdeFichero();
    }
	
	/**
	  * Loads results from a set of instances contained in the supplied
	  * file.
	  */
    private void instanciasDesdeFichero(){
		
		try{
			
			File f = new File(this.filepath);
			Instances myInsts = null;
			Resample r = null;
			double size_sample = 0;
			
			//Si se trata de un archivo .arff
			if (f.getName().toLowerCase().endsWith(Instances.FILE_EXTENSION)){
				
				//Se lee el archivo ARFF.
				BufferedReader datos = new BufferedReader(new FileReader(f));
				
				//Se instancian los ejemplos.
				//this.instancias = new Instances(datos);
				
				//Resample.
				myInsts = new Instances(datos);

				//
				datos.close();
				
			//Si se trata de un archivo .csv	
			}else if(f.getName().toLowerCase().endsWith(CSVLoader.FILE_EXTENSION)){				
				
				//
				CSVLoader loader = new CSVLoader();
				loader.setSource(f);
				//this.instancias = loader.getDataSet();
				
				//Resample.
				myInsts = loader.getDataSet();
				
			}else{
				
				throw new Exception("El tipo de archivo debe ser .arff o .csv!");
			}
			
			r = new Resample(); 
			r.setNoReplacement(false);
			
			//System.out.println("NUM INSTANCES: " + myInsts.numInstances());
				
			size_sample = (double) Math.round(((double)(this.max_instancias * 100) / (double) myInsts.numInstances()) * 100) / 100;
			//size_sample = Double.parseDouble(String.format("%.2f", (double)((double)(this.max_instancias * 100) / (double) myInsts.numInstances())));
			
			//System.out.println("SIZE SAMPLE: " + size_sample);
			
			myInsts.setClassIndex(myInsts.numAttributes() - 1);
				
			r.setSampleSizePercent(size_sample); // or whatever % you require 
			r.setInputFormat(myInsts);
			this.instancias = Filter.useFilter(myInsts, r); 

			//Se ramdomizan las filas de ejemplos.
			this.instancias.randomize(new Random(seed));
			
			//System.out.println("INSTANCIAS!!!!!!!!!!!!!!: " + this.instancias.numInstances());
			//System.out.println(this.instancias);
			
		}catch(Exception e){
			System.err.println("Este es el problema: " + e);
			e.printStackTrace();
		}
    }
	
	
	/** Se obtienen los conjuntos de entrenamiento y de testeo, a partir del conjunto de instancias.
	  * @param randomizar Entero que puede ser 0 o 1. Un 1 indica que hay que randomizar las ubicaciones.
	  * @param porc_split Double que debe ser mayor a 0 y menor 1. Indica qué porcentaje de las instancias es para ENTRENAMIENTO.
	  * @return No devuelve ningun valor.
	  * @throws No dispara ninguna excepcion.
	  */
    public void trainTestInstancias(int randomizar, double porc_split){
		
		try{
			
			//Se ramdomiza la ubicacion de las instancias.
			if (randomizar == 1){
				
				this.instancias.randomize(new Random(this.seed));				
			}

			//Se realiza el split, se obtienen dos subconjunto: train y test.
			if (porc_split > 0 && porc_split < 1){
				
				this.trainSize = (int) Math.round(this.instancias.numInstances() * porc_split);				
			}else{
				
				this.trainSize = (int) Math.round(this.instancias.numInstances() * this.porcentaje_split);
			}
			
			//
			this.testSize = this.instancias.numInstances() - this.trainSize;
			this.train = new Instances(this.instancias, 0, this.trainSize);
			this.test = new Instances(this.instancias, this.trainSize, this.testSize);
			
			//this.test1 = new Instances(this.instancias, this.trainSize, this.extraSize);
			//this.test2 = new Instances(this.instancias, this.trainSize, this.extraSize);
			
			//Se configura el atributo class.
			this.instancias.setClassIndex(this.instancias.numAttributes() - 1);
			this.train.setClassIndex(this.train.numAttributes() - 1);
			this.test.setClassIndex(this.test.numAttributes() - 1);
					
		}catch(Exception e){
			System.err.println("Error en: " + this.filename);
			System.err.println("Exception: " + e);
			//e.printStackTrace();
		}
    }
	
	/** Se obtienen los conjuntos de entrenamiento y de testeo, a partir del conjunto de instancias.
	  * @param randomizar Entero que puede ser 0 o 1. Un 1 indica que hay que randomizar las ubicaciones.
	  * @param porc_split Double que debe ser mayor a 0 y menor 1. Indica qué porcentaje de las instancias es para entrenamiento.
	  * @return No devuelve ningun valor.
	  * @throws No dispara ninguna excepcion.
	  */
    public List<LinkedHashMap<String,String>> construirYEvaluar(String tipoEvaluacion, int k){
		
		List<LinkedHashMap<String,String>> lista_diccionarios = new ArrayList<LinkedHashMap<String,String>>();
		String name = "";
		StringWriter errores = null;
		List<String> etiquetas_actuales = new ArrayList<String>();
		boolean bandera = false;
		
		try{
			
			this.fstream = new FileWriter("error_log",true);
			this.out = new BufferedWriter(fstream);
		
		}catch(Exception e){}
		
		try{
			
			Classifier modelo;
			Evaluation evaluar;
			Attribute atributo_de_clase = this.instancias.classAttribute();
			int cantidad_etiquetas = atributo_de_clase.numValues();
			String etiqueta;
			DecimalFormat df = new DecimalFormat("#.##");
			int indice_clase = 0;
			
			//
			double numero_instancias = 0;
			double cantidad_aciertos = 0;
			double porcentaje_aciertos = 0;
			double cantidad_desaciertos = 0;
			double porcentaje_desaciertos = 0;
			double kappa_statistic = 0;
			double mean_absolute_error = 0;
			double relative_absolute_error = 0;
			double root_mean_squared_error = 0;
			double root_relative_squared_error = 0;
			
			//
			double true_positive_rate = 0;
			double false_positive_rate = 0;
			double true_negative_rate = 0;
			double false_negative_rate = 0;
			double precision = 0;
			double recall = 0;
			double f_measure = 0;
			double mcc = 0;
			double roc_area = 0;
			double prc_area = 0;
			double[][] confusion_matrix = null;
			
		
			//Conjunto de clasificadores.
			Classifier[] modelos = {
				//BAYES.			
				new BayesNet(),
				new NaiveBayes(),
				new NaiveBayesUpdateable(),
				//FUNCTIONS.			
				new Logistic(),
				new MultilayerPerceptron(),
				new SimpleLogistic(),
				new SMO(),
				//RULES.
				new OneR(),
				new DecisionTable(),
				new JRip(),
				new PART(),
				new ZeroR(),
				//TREES.			
				new DecisionStump(),
				new J48(),
				new LMT(),
				new RandomForest(),
				new RandomTree(),
				new REPTree(),
			};
			
			if (k > 1){
				this.folds = k;
			}
			
			//Etiquetas.
			for (int m=0; m < cantidad_etiquetas; m++){
					
				etiqueta = atributo_de_clase.value(m);
				etiquetas_actuales.add(etiqueta);
			}
			
			bandera = false;
			
			//Por cada modelo, se construye su clasificador y se evalua.
			for (int j=0; j<modelos.length; j++){
				
				//Se construye un diccionario por cada modelo, que contiene las metricas obtenidas.
				LinkedHashMap<String,String> diccionario_actual = new LinkedHashMap<>();
				lista_diccionarios.add(diccionario_actual);
				
				//Se construye el clasificador.
				modelo = modelos[j];
				name = modelo.getClass().getName();
				diccionario_actual.put("modelo", name);
				
				//Guardamos el modelo.
				//SerializationHelper.write(name, modelo);
				
				//Cargamos un modelo
				//Classifier cargarModelo = (Classifier) SerializationHelper.read(nombre);
				
				//Se evalua el modelo.
				if (tipoEvaluacion == "split"){
					
					try{
					
						modelo.buildClassifier(this.train);
						//constructor Evaluation.Evaluation(Instances,CostMatrix)
						evaluar = new Evaluation(this.test);
						evaluar.evaluateModel(modelo, this.test);
					
					}catch(Exception e){
						
						try{
							errores = new StringWriter();
							this.out.write("Archivo que dio error: " + this.filename + "\n");
							this.out.write("Modelo que dio error: " + name + "\n");
							this.out.write("Evaluacion que dio error: " + tipoEvaluacion + "\n");
							this.out.write("Exception: " + "\n");
							e.printStackTrace(new PrintWriter(errores));
							this.out.write(errores.toString() + "\n");
						}catch(Exception ex){}
						
						continue;
					}
					
				}else if (tipoEvaluacion == "cross"){
					
					try{
					
						modelo.buildClassifier(this.train);
						evaluar = new Evaluation(this.test);
						evaluar.crossValidateModel(modelo, this.test, this.folds, new Random(this.seed));
					
					}catch(Exception e){
						
						try{
							errores = new StringWriter();
							this.out.write("Archivo que dio error: " + this.filename + "\n");
							this.out.write("Modelo que dio error: " + name + "\n");
							this.out.write("Evaluacion que dio error: " + tipoEvaluacion + "\n");
							this.out.write("Exception: " + "\n");
							e.printStackTrace(new PrintWriter(errores));
							this.out.write(errores.toString() + "\n");
						}catch(Exception ex){}
						
						continue;
					}
					
				}else if (tipoEvaluacion == "crossEstratificado"){
					
					try{
					
						modelo.buildClassifier(this.instancias);
						evaluar = new Evaluation(this.instancias);
						evaluar.crossValidateModel(modelo, this.instancias, this.folds, new Random(this.seed));
					
					}catch(Exception e){
						
						try{
							errores = new StringWriter();
							this.out.write("Archivo que dio error: " + this.filename + "\n");
							this.out.write("Modelo que dio error: " + name + "\n");
							this.out.write("Evaluacion que dio error: " + tipoEvaluacion + "\n");
							this.out.write("Exception: " + "\n");
							e.printStackTrace(new PrintWriter(errores));
							this.out.write(errores.toString() + "\n");
						
						}catch(Exception ex){}
						
						continue;
					}
					
				}else{
					
					System.out.println("Tipo de evaluacion desconocida!");
					return null;
				}
				
				bandera = true;
				
				//INDICADORES RESUMEN O GLOBALES.
				
				//Numero de instancias.
				numero_instancias = evaluar.numInstances();
				
				//Aciertos.
				cantidad_aciertos = evaluar.correct();
				porcentaje_aciertos = evaluar.pctCorrect();
				
				//Desaciertos.
				cantidad_desaciertos = evaluar.incorrect();
				porcentaje_desaciertos = evaluar.pctIncorrect();
				//Kappa.
				kappa_statistic = evaluar.kappa();
				
				//Error absoluto promedio.
				mean_absolute_error = evaluar.meanAbsoluteError();
				
				//Error relativo absoluto.
				relative_absolute_error = evaluar.relativeAbsoluteError();

				//Raiz del error promedio cuadratico.
				root_mean_squared_error = evaluar.rootMeanSquaredError();
				
				//Raiz del error relativo cuadratico.
				root_relative_squared_error = evaluar.rootRelativeSquaredError();
				
				//Se carga en el diccionario.
				//System.out.println(Double.toString(Double.valueOf(df.format(numero_instancias))));
				//numero_instancias
				diccionario_actual.put("NINS", String.format("%.2f", numero_instancias));
				//porcentaje_aciertos
				diccionario_actual.put("PACI", String.format("%.2f", porcentaje_aciertos));
				//porcentaje_desaciertos
				diccionario_actual.put("PDES", String.format("%.2f", porcentaje_desaciertos));
				//kappa_statistic
				diccionario_actual.put("KSTA", String.format("%.2f", kappa_statistic));
				//mean_absolute_error
				diccionario_actual.put("MAERR", String.format("%.2f", mean_absolute_error));
				//relative_absolute_error
				diccionario_actual.put("RAERR", String.format("%.2f", relative_absolute_error));
				//root_mean_squared_error
				diccionario_actual.put("RMSERR", String.format("%.2f", root_mean_squared_error));
				//root_relative_squared_error
				diccionario_actual.put("RRSERR", String.format("%.2f", root_relative_squared_error));				
				
				//INDICADORES DE PRECISION POR CLASE.
				for (int q=0; q < this.etiquetas.size(); q++){
					
					etiqueta = this.etiquetas.get(q);
					
					if (etiquetas_actuales.contains(etiqueta)){
						
						indice_clase = atributo_de_clase.indexOfValue(etiqueta);
						
						//Verdaderos positivos.
						true_positive_rate = evaluar.truePositiveRate(indice_clase);
						
						//Falsos positivos.
						false_positive_rate = evaluar.falsePositiveRate(indice_clase);
						
						//Verdaderos negativos.
						true_negative_rate = evaluar.trueNegativeRate(indice_clase);
						
						//Falsos negativos.
						false_negative_rate = evaluar.falseNegativeRate(indice_clase);
						
						//Precision.
						precision = evaluar.precision(indice_clase);
						
						//Recall.
						recall = evaluar.recall(indice_clase);
						
						//F-Measure.
						f_measure = evaluar.fMeasure(indice_clase);
						
						//Matthews Correlation Coefficient (sometimes called phi coefficient).
						mcc = evaluar.matthewsCorrelationCoefficient(indice_clase);
						
						//Area under ROC.
						roc_area = evaluar.areaUnderROC(indice_clase);
						
						//Area under precision-recall curve (AUPRC).
						prc_area = evaluar.areaUnderPRC(indice_clase);
						
					}else{
						
						//Verdaderos positivos.
						true_positive_rate = 0;
						
						//Falsos positivos.
						false_positive_rate = 0;
						
						//Verdaderos negativos.
						true_negative_rate = 0;
						
						//Falsos negativos.
						false_negative_rate = 0;
						
						//Precision.
						precision = 0;
						
						//Recall.
						recall = 0;
						
						//F-Measure.
						f_measure = 0;
						
						//Matthews Correlation Coefficient (sometimes called phi coefficient).
						mcc = 0;
						
						//Area under ROC.
						roc_area = 0;
						
						//Area under precision-recall curve (AUPRC).
						prc_area = 0;
					}
					
					//Se carga en el diccionario.
					//true_positive_rate
					diccionario_actual.put(etiqueta + "_" + "TPRA", String.format("%.2f", true_positive_rate));
					//false_positive_rate
					diccionario_actual.put(etiqueta + "_" + "FPRA", String.format("%.2f", false_positive_rate));
					//true_negative_rate
					diccionario_actual.put(etiqueta + "_" + "TNRA", String.format("%.2f", true_negative_rate));
					//false_negative_rate
					diccionario_actual.put(etiqueta + "_" + "FNRA", String.format("%.2f", false_negative_rate));
					//precision
					diccionario_actual.put(etiqueta + "_" + "PREC", String.format("%.2f", precision));
					//recall
					diccionario_actual.put(etiqueta + "_" + "RCALL", String.format("%.2f", recall));
					//f_measure
					diccionario_actual.put(etiqueta + "_" + "FMEA", String.format("%.2f", f_measure));
					//mcc
					diccionario_actual.put(etiqueta + "_" + "MCC", String.format("%.2f", mcc));
					//roc_area
					diccionario_actual.put(etiqueta + "_" + "ROCA", String.format("%.2f", roc_area));
					//prc_area
					diccionario_actual.put(etiqueta + "_" + "PRCA", String.format("%.2f", prc_area));					
				}

				//MATRIZ DE CONFUSION.
				//confusion_matrix = evaluar.confusionMatrix();
				
				//for (int row=0; row < confusion_matrix.length; row++){
				//	for (int col=0; col < confusion_matrix[row].length; col++){
				//		System.out.print(confusion_matrix[row][col] + " ");
				//	}
				//	System.out.println();
				//}
				
				//System.out.println();
			}
		
		}catch(Exception e){
			
			try{
				errores = new StringWriter();
				this.out.write("Archivo no procesado: " + this.filename + "\n");
				this.out.write("Modelo que dio error: " + name + "\n");
				this.out.write("Evaluacion que dio error: " + tipoEvaluacion + "\n");
				this.out.write("Exception: " + "\n");
				e.printStackTrace(new PrintWriter(errores));
				this.out.write(errores.toString() + "\n");
			
			}catch(Exception ex){}
			
			return null;
		}
		
		//Se carga la cabecera actual.
		this.extra_cabecera = "tipo_evaluacion" + ";" + "producto_evaluado" + ";";
		LinkedHashMap<String,String> diccionario_keys = lista_diccionarios.get(1);
		this.lista_claves = new ArrayList<String>();
		
		//Se obtienen las claves.
		for (String key:diccionario_keys.keySet()){
									
			this.lista_claves.add(key);
			this.cabecera = this.cabecera + key + ";";
		}
								
		this.cabecera = this.cabecera + this.extra_cabecera;
		this.extra_datos = tipoEvaluacion + ";" + this.filename + ";";
		
		try{
			
			this.out.close();
			
		}catch(Exception e){}
		
		
		if (bandera == false){
			
			return null;
		}
		
		return lista_diccionarios;
    }//Hasta aqui construirYEvaluar.
	
	
	public String obtenerCabecera(){
		
		return this.cabecera;
	
	}
	
	public String obtenerExtraDatos(){
		
		return this.extra_datos;
	
	}
	
	public List<String> obtenerListaClaves(){
		
		return this.lista_claves;
		
	}
	
}//Cierre de la clase BIML.