/*
  Compilar con weka.jar, v3.8+, en el mismo directorio y 
  $ javac -Xlint -cp .:weka.jar DataParser.java rating.list
  (en linux)
 */

import java.util.List;
import java.util.ArrayList;
import java.util.regex.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.converters.ArffSaver;

public class DataParser{
    private final String dataTitle = "MOVIE RATINGS REPORT";
    private final String dataPattern = "^\\s+.*\\s+\\d\\s+(?<rating>[\\d\\.]+)\\s+\"(?<name>.+)\"\\s+\\((?<year>\\d+)\\)\\s+(?<extra>.+)$";
    private final String ENCODING = "latin1";
    private String inputFilename;
    private BufferedReader buffer;
    private Pattern dataRegex;
    private Pattern nameRegex;

    // Datos a utilizar durante todo el procedimiento
    private Instances dataset;
    private ArrayList<Attribute> attributes;

    public DataParser(String inputFilename){
	this.inputFilename = inputFilename;
	this.dataRegex     = Pattern.compile(dataPattern);

	// Creamos atributo numerico
	Attribute id = new Attribute("ID");
	// Creamos atributo de cadena
	Attribute fullname = new Attribute("Fullname", (List<String>) null);
	Attribute name = new Attribute("Name", (List<String>) null);
	Attribute rating = new Attribute("Rating");
	Attribute year = new Attribute("Year");

	// Listamos atributos para la definición del dataset
	attributes = new ArrayList<>();
	attributes.add(id);
	attributes.add(fullname);
	attributes.add(name);
	attributes.add(rating);
	attributes.add(year);

	// Dataset inicializado para 50000 instancias
	dataset = new Instances("Movies Rating Dataset", attributes, 50000);
    }
        
    private void readFile() throws IOException{
	File file = new File( inputFilename );
	FileInputStream fis = new FileInputStream( file );
	InputStreamReader isr = new InputStreamReader( fis, ENCODING );
	
	buffer = new BufferedReader( isr );
    }

    public void parse() throws IOException{
	readFile();
	
	while( !buffer.readLine().equals( dataTitle ) );
	buffer.readLine();
	buffer.readLine();

	int id = 0;

	String line = buffer.readLine();
	while( line != null ){
	    Matcher m = dataRegex.matcher(line);
	    if( m.find() ){
		Double rating = Double.parseDouble( m.group("rating") );
		String name = m.group("name");
		int year = Integer.parseInt( m.group("year") );
		String extra = m.group("extra");
		String fullname = "\"" + name + "\" (" + year + ") " + extra;

		// Creamos una instancia de 5 atributos
		Instance instance = new DenseInstance(5);
		// Especificamos los valores utilizando instancias de weka.core.Attribute
		// utilizados en la definición del dataset
		instance.setValue(attributes.get(0), id++);
		instance.setValue(attributes.get(1), fullname);
		instance.setValue(attributes.get(2), name);
		instance.setValue(attributes.get(3), rating);
		instance.setValue(attributes.get(4), year);

		// Agregamos al dataset
		dataset.add( instance );
	    }
	    line = buffer.readLine();
	}
	
	buffer.close();
    }

    public void save(String path) throws IOException{
	// Guardamos al formato ARFF
	// Se puede reemplazar por otro conversor modificando la siguiente linea
	ArffSaver saver = new ArffSaver();
	
	saver.setInstances( dataset );
	saver.setFile( new File(path) );
	saver.writeBatch();
    }
    
    public static void main(String[] args){
	try{
	    String inputFilename = args[0];
	    DataParser parser = new DataParser(inputFilename);
	    parser.parse();
	    parser.save("./ratings.arff");
	}catch(ArrayIndexOutOfBoundsException | IOException e){
	    System.err.println("Ocurrió un error!");
	    System.err.println(e.getMessage());
	}
    }
}
