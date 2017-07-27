import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.trees.J48;


public class biml{
	public static void main(String[] arg){
		try {
			String archivo = "C:\\Users\\raul\\Desktop\\Weka_Practicas\\1ra_Codificacion\\data.arff"; 
			BufferedReader reader = new BufferedReader(new FileReader(archivo));
			Instances data = new Instances(reader);
			reader.close();
			//Setting class attribute.
			data.setClassIndex(data.numAttributes() - 1);
			
			String[] options = new String[1];
			options[0] = "-U";            // unpruned tree
			J48 tree = new J48();         // new instance of tree
			tree.setOptions(options);     // set the options
			tree.buildClassifier(data);   // build classifier
			
			
		}catch(Exception e){};
	}
}