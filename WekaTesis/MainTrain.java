import Clases.BIML;
import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;

//OK!---
//javac -cp ".\weka.jar;." MainTrain.java .\Clases\*.java
//java -cp ".\weka.jar;." MainTrain
//------

//javac -cp weka.jar MainTrain.java .\Clases\*.java
//java -cp . MainTrain

//javac -cp "C:\Program Files\Weka-3-8\weka.jar;." MainTrain.java .\Clases\*.java
//java -cp "C:\Program Files\Weka-3-8\weka.jar;." MainTrain

//java -cp weka.jar weka.core.converters.CSVLoader datos.csv > datos.arff

public class MainTrain {
	public static void main(String[] s){
		
		BIML b;
		int randomizar;
		double porc_split;
		int k;
		int max_instancias;
		List<String> lista_paths;
		List<String> lista_evaluaciones;
		String evaluacion;
		String path;
		List<String> etiquetas;
		List<LinkedHashMap<String,String>> lista_diccionarios_clasificadores;
		List<String> lista_claves;
		LinkedHashMap<String,String> diccionario_actual;
		String archivo_datos;
		String archivo_evaluacion;
		FileWriter fstream;
		BufferedWriter out;
		String clave;
		String valor;
		String cabecera;
		String fila;
		String extra_cabecera;
		String extra_datos;
		
		try{
					
			//
			long startTime = System.currentTimeMillis();
			randomizar = 1;
			porc_split = 0.6;
			k = 10;
			max_instancias = 151;

			lista_paths = new ArrayList<String>();
			//lista_paths.add("KPI/MENSUAL/");
			//lista_paths.add("KPI/QUINCENAL/");
			lista_paths.add("KPI/SEMANAL/");

			lista_evaluaciones = new ArrayList<String>();
			lista_evaluaciones.add("split");
			//lista_evaluaciones.add("cross");
			lista_evaluaciones.add("crossEstratificado");
			
			etiquetas = new ArrayList<String>();
			etiquetas.add("Nada");
			etiquetas.add("Poco");
			etiquetas.add("Medio");
			etiquetas.add("Mucho");
			
			int bandera = 0;
			int contador = 0;
			
			//Se recorre cada path para validar los archivos CSV.
			for (int p=0; p<lista_paths.size(); p++){
				
				path = lista_paths.get(p);
				contador = 0;
				
				System.out.println("Verificando....  " + path);
				
				File folder = new File("./Generados/" + path);
				File[] listOfFiles = folder.listFiles();

				for (File file:listOfFiles){
					if (file.isFile()){
						archivo_datos = file.getName();
						b = new BIML(path + archivo_datos, etiquetas, max_instancias);
						contador++;
						System.out.println(archivo_datos);
					}
				}
			}
			
			System.out.println(contador + " archivos procesados...");
			
			//System.exit(0);
			//System.gc();
			//System.runFinalization();
			
			contador = 0;
			
			//Se recorre cada path para crear los clasificadores.
			for (int p=0; p<lista_paths.size(); p++){
				
				path = lista_paths.get(p);
				
				System.out.println("Recorriendo....  " + path);
				
				File folder = new File("./Generados/" + path);
				File[] listOfFiles = folder.listFiles();

				for (File file:listOfFiles){
					if (file.isFile()){
						
						archivo_datos = file.getName();
						
						//
						b = new BIML(path + archivo_datos, etiquetas, max_instancias);
						contador++;
						
						//
						//System.out.println("Creando instancias....");
						b.trainTestInstancias(randomizar,porc_split);
						
						for (int e=0; e<lista_evaluaciones.size(); e++){
							
							evaluacion = lista_evaluaciones.get(e);
							
							//
							//System.out.println("Construyendo los clasificadores....");
							lista_diccionarios_clasificadores = b.construirYEvaluar(evaluacion,k);
							
							if(lista_diccionarios_clasificadores == null){
								
								continue;
							}
							
							archivo_evaluacion = evaluacion + ".csv";
							fstream = new FileWriter(archivo_evaluacion,true);
							out = new BufferedWriter(fstream);
							lista_claves = b.obtenerListaClaves();
							
							//
							if(bandera < lista_evaluaciones.size()){
								
								//
								cabecera = b.obtenerCabecera()  + "\n";
								out.write(cabecera);
								bandera++;
							}
							
							extra_datos = b.obtenerExtraDatos();
							
							//
							for (int i=0; i<lista_diccionarios_clasificadores.size(); i++){
								
								diccionario_actual = lista_diccionarios_clasificadores.get(i);
								fila = "";
								
								for (int j=0; j<lista_claves.size(); j++){
									
									valor = diccionario_actual.get(lista_claves.get(j));
									fila = fila + valor + ";";
								}
								
								if (fila != ""){
									
									//
									fila = fila + extra_datos + "\n";
									out.write(fila);									
								}
							}
							
							out.close();
						}
						
						//Luego de procesar cada archivo.
						System.gc();
						System.runFinalization();
						
						System.out.println(contador + " archivos");
						
					}
				}
			}
			
			//
			long estimatedTime = System.currentTimeMillis() - startTime;
			System.out.println("Tiempo de ejecucion: " + TimeUnit.MILLISECONDS.toSeconds(estimatedTime) + " segundos...");
		
    	}catch(Exception e){
    		e.printStackTrace();
    	}
	}
}