package us.jameshoward.iristypes;

import java.util.Hashtable;

public class IrisDriver {
	
	public IrisDriver() { }
	
	public static void main(String[] args) throws Exception {
		Hashtable<String, String> values = new Hashtable<String, String>();
		Iris irisModel = new Iris();

		for(int i = 0; i < args.length; i++) {
 			String[] tokens = args[i].split("=");
			
			values.put(tokens[0], tokens[1]);
		}
		
		System.out.println("Classification: " + irisModel.classifySpecies(values));
	}
	
}
